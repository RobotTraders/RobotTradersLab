from datetime import UTC, datetime, timedelta

import pytest

from robottraderslab._core import Execution, OrderSide, Symbol
from robottraderslab.analyser.execution_equity import (
    build_equity_curve,
    recover_opening_balance,
)

_BTC = Symbol.create("BTC/USDT:USDT")


def _execution(**overrides: object) -> Execution:
    arguments = {
        "execution_id": "execution-1",
        "order_id": "order-1",
        "symbol": _BTC,
        "side": OrderSide.SELL,
        "price": 100.0,
        "quantity": 1.0,
        "timestamp": datetime(2026, 8, 2, 12, tzinfo=UTC),
        "realised_profit": 0.0,
        "fee": 0.0,
    }
    arguments.update(overrides)
    return Execution(**arguments)  # type: ignore[arg-type]


class TestNoExecutions:
    def test_the_curve_stays_flat_at_zero(self):
        since = datetime(2026, 8, 1, 10, tzinfo=UTC)
        until = datetime(2026, 8, 3, 15, tzinfo=UTC)

        curve = build_equity_curve([], since=since, until=until, opening_balance=None)

        assert (curve == 0.0).all()

    def test_a_window_shorter_than_a_day_carries_only_its_two_edges(self):
        since = datetime(2026, 8, 1, 10, tzinfo=UTC)
        until = datetime(2026, 8, 1, 14, tzinfo=UTC)

        curve = build_equity_curve([], since=since, until=until, opening_balance=None)

        assert len(curve) == 2
        assert list(curve.index) == [since, until]


class TestCheckpoints:
    @pytest.fixture
    def since(self) -> datetime:
        return datetime(2026, 8, 1, 10, tzinfo=UTC)

    @pytest.fixture
    def until(self) -> datetime:
        return datetime(2026, 8, 3, 15, tzinfo=UTC)

    def test_every_midnight_the_window_covers_is_a_checkpoint(self, since, until):
        curve = build_equity_curve([], since=since, until=until, opening_balance=None)

        assert datetime(2026, 8, 2, tzinfo=UTC) in curve.index
        assert datetime(2026, 8, 3, tzinfo=UTC) in curve.index

    def test_the_windows_own_edges_are_checkpoints(self, since, until):
        curve = build_equity_curve([], since=since, until=until, opening_balance=None)

        assert since in curve.index
        assert until in curve.index

    def test_checkpoints_are_ordered_by_time(self, since, until):
        curve = build_equity_curve([], since=since, until=until, opening_balance=None)

        assert list(curve.index) == sorted(curve.index)


class TestBookedAmountsAccumulate:
    @pytest.fixture
    def since(self) -> datetime:
        return datetime(2026, 8, 1, 10, tzinfo=UTC)

    @pytest.fixture
    def until(self) -> datetime:
        return datetime(2026, 8, 3, 15, tzinfo=UTC)

    @pytest.fixture
    def executions(self) -> list[Execution]:
        return [
            _execution(
                timestamp=datetime(2026, 8, 2, 12, tzinfo=UTC),
                realised_profit=50.0,
                fee=5.0,
            ),
            _execution(
                timestamp=datetime(2026, 8, 3, 6, tzinfo=UTC),
                realised_profit=-20.0,
                fee=2.0,
            ),
        ]

    def test_a_checkpoint_before_every_execution_sits_at_zero(
        self, since, until, executions
    ):
        curve = build_equity_curve(
            executions, since=since, until=until, opening_balance=None
        )

        assert curve[since] == 0.0

    def test_a_checkpoint_only_carries_what_was_booked_up_to_it(
        self, since, until, executions
    ):
        curve = build_equity_curve(
            executions, since=since, until=until, opening_balance=None
        )

        assert curve[datetime(2026, 8, 3, tzinfo=UTC)] == pytest.approx(45.0)

    def test_the_final_checkpoint_carries_everything_booked(
        self, since, until, executions
    ):
        curve = build_equity_curve(
            executions, since=since, until=until, opening_balance=None
        )

        assert curve[until] == pytest.approx(23.0)

    def test_an_unstated_profit_or_fee_is_treated_as_zero(self, since, until):
        executions = [
            _execution(
                timestamp=datetime(2026, 8, 2, 12, tzinfo=UTC),
                realised_profit=None,
                fee=None,
            )
        ]

        curve = build_equity_curve(
            executions, since=since, until=until, opening_balance=None
        )

        assert curve[until] == 0.0


class TestTheCurveTurnsWhereItWasBooked:
    def test_a_window_shorter_than_a_day_carries_its_executions(self):
        since = datetime(2026, 8, 1, 9, 0, tzinfo=UTC)
        until = datetime(2026, 8, 1, 15, 0, tzinfo=UTC)
        booked = [
            _execution(timestamp=since + timedelta(hours=1), realised_profit=100.0),
            _execution(timestamp=since + timedelta(hours=3), realised_profit=-90.0),
        ]

        curve = build_equity_curve(
            booked, since=since, until=until, opening_balance=None
        )

        assert list(curve) == [0.0, 100.0, 10.0, 10.0]

    def test_a_peak_between_two_midnights_stays_on_the_curve(self):
        since = datetime(2026, 8, 1, tzinfo=UTC)
        until = datetime(2026, 8, 3, tzinfo=UTC)
        booked = [
            _execution(
                timestamp=datetime(2026, 8, 1, 12, tzinfo=UTC), realised_profit=50.0
            ),
            _execution(
                timestamp=datetime(2026, 8, 1, 18, tzinfo=UTC), realised_profit=-50.0
            ),
        ]

        curve = build_equity_curve(
            booked, since=since, until=until, opening_balance=None
        )

        assert curve.max() == pytest.approx(50.0)

    def test_an_execution_moment_appears_once_even_at_a_midnight(self):
        since = datetime(2026, 8, 1, tzinfo=UTC)
        until = datetime(2026, 8, 3, tzinfo=UTC)
        booked = [
            _execution(timestamp=datetime(2026, 8, 2, tzinfo=UTC), realised_profit=10.0)
        ]

        curve = build_equity_curve(
            booked, since=since, until=until, opening_balance=None
        )

        assert curve.index.is_unique


class TestTheCurveOpensOnAStatedBalance:
    @pytest.fixture
    def since(self) -> datetime:
        return datetime(2026, 8, 1, 10, tzinfo=UTC)

    @pytest.fixture
    def until(self) -> datetime:
        return datetime(2026, 8, 1, 15, tzinfo=UTC)

    def test_a_window_with_nothing_booked_holds_the_balance_throughout(
        self, since, until
    ):
        curve = build_equity_curve([], since=since, until=until, opening_balance=500.0)

        assert (curve == 500.0).all()

    def test_every_point_is_the_balance_plus_what_was_booked_up_to_it(
        self, since, until
    ):
        booked = [
            _execution(timestamp=since + timedelta(hours=1), realised_profit=100.0),
            _execution(timestamp=since + timedelta(hours=3), realised_profit=-40.0),
        ]

        curve = build_equity_curve(
            booked, since=since, until=until, opening_balance=500.0
        )

        assert list(curve) == [500.0, 600.0, 560.0, 560.0]

    def test_a_loss_before_any_gain_leaves_the_peak_at_the_balance(self, since, until):
        booked = [
            _execution(timestamp=since + timedelta(hours=1), realised_profit=-40.0)
        ]

        curve = build_equity_curve(
            booked, since=since, until=until, opening_balance=500.0
        )

        assert curve.cummax().min() == pytest.approx(500.0)


class TestRecoveringTheOpeningBalance:
    def test_the_windows_profit_comes_back_off_what_the_account_holds_now(self):
        booked = [_execution(realised_profit=50.0, fee=5.0)]

        opening_balance = recover_opening_balance(545.0, booked)

        assert opening_balance == pytest.approx(500.0)

    def test_a_window_with_nothing_booked_opened_on_what_it_holds_now(self):
        opening_balance = recover_opening_balance(500.0, [])

        assert opening_balance == pytest.approx(500.0)

    def test_a_loss_across_the_window_means_it_opened_on_more(self):
        booked = [_execution(realised_profit=-120.0, fee=5.0)]

        opening_balance = recover_opening_balance(375.0, booked)

        assert opening_balance == pytest.approx(500.0)

    @pytest.mark.parametrize(
        ("current_balance", "profit"),
        [(50.0, 50.0), (10.0, 60.0)],
        ids=["nothing_left", "more_earned_than_held"],
    )
    def test_a_balance_the_arithmetic_leaves_nothing_positive_of(
        self, current_balance, profit
    ):
        booked = [_execution(realised_profit=profit, fee=0.0)]

        opening_balance = recover_opening_balance(current_balance, booked)

        assert opening_balance is None
