from datetime import UTC, datetime

import pandas as pd
import pytest

from robottraderslab._core import Execution, OrderSide, Symbol
from robottraderslab.analyser.execution_trades import build_trades
from robottraderslab.exchanges import client_order_id_carrying

_BTC = Symbol.create("BTC/USDT:USDT")
_ETH = Symbol.create("ETH/USDT:USDT")
_SOL = Symbol.create("SOL/USDT:USDT")
_SINCE = datetime(2026, 1, 1, tzinfo=UTC)
_UNTIL = datetime(2027, 1, 1, tzinfo=UTC)


def _execution(**overrides: object) -> Execution:
    arguments = {
        "execution_id": "execution-1",
        "order_id": "order-1",
        "symbol": _BTC,
        "side": OrderSide.BUY,
        "price": 100.0,
        "quantity": 1.0,
        "timestamp": datetime(2026, 8, 1, tzinfo=UTC),
        "kind": "market",
    }
    arguments.update(overrides)
    return Execution(**arguments)  # type: ignore[arg-type]


class TestNoTrades:
    def test_no_executions_yields_no_trades(self):
        paired = build_trades([], since=_SINCE, until=_UNTIL)

        assert paired.trades.empty
        assert paired.unpairable == 0

    def test_only_opening_executions_yield_no_trades(self):
        paired = build_trades(
            [_execution(realised_profit=None), _execution(realised_profit=0.0)],
            since=_SINCE,
            until=_UNTIL,
        )

        assert paired.trades.empty


class TestClosingAPosition:
    @pytest.fixture
    def trades(self) -> pd.DataFrame:
        opened = _execution(
            side=OrderSide.BUY,
            price=100.0,
            quantity=2.0,
            fee=1.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            kind="market",
            realised_profit=None,
        )
        closed = _execution(
            side=OrderSide.SELL,
            price=110.0,
            quantity=2.0,
            fee=1.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            kind="take-profit",
            realised_profit=20.0,
        )
        return build_trades(
            [opened, closed],
            since=_SINCE,
            until=_UNTIL,
        ).trades

    def test_a_full_close_yields_one_trade(self, trades):
        assert len(trades) == 1

    def test_the_trade_carries_the_entry_and_exit_prices(self, trades):
        assert trades.iloc[0]["entry_price"] == 100.0
        assert trades.iloc[0]["exit_price"] == 110.0

    def test_the_trade_carries_the_entry_and_exit_times(self, trades):
        assert trades.iloc[0]["entry_time"] == datetime(2026, 8, 1, tzinfo=UTC)
        assert trades.iloc[0]["exit_time"] == datetime(2026, 8, 2, tzinfo=UTC)

    def test_a_buy_then_sell_is_a_long(self, trades):
        assert trades.iloc[0]["side"] == "long"

    def test_a_sell_then_buy_is_a_short(self):
        opened = _execution(
            side=OrderSide.SELL, quantity=2.0, fee=1.0, realised_profit=None
        )
        closed = _execution(
            side=OrderSide.BUY, quantity=2.0, fee=1.0, realised_profit=15.0
        )

        trades = build_trades(
            [opened, closed],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert trades.iloc[0]["side"] == "short"

    def test_the_gross_pnl_is_the_figure_the_venue_booked(self, trades):
        assert trades.iloc[0]["gross_pnl"] == 20.0

    def test_the_net_pnl_subtracts_both_fees(self, trades):
        assert trades.iloc[0]["net_pnl"] == pytest.approx(18.0)

    def test_the_reasons_come_from_each_executions_kind(self, trades):
        assert trades.iloc[0]["entry_reason"] == "market"
        assert trades.iloc[0]["exit_reason"] == "take-profit"

    def test_an_unstated_fee_is_treated_as_zero(self):
        opened = _execution(
            side=OrderSide.BUY, quantity=1.0, fee=None, realised_profit=None
        )
        closed = _execution(
            side=OrderSide.SELL, quantity=1.0, fee=None, realised_profit=5.0
        )

        trades = build_trades(
            [opened, closed],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert trades.iloc[0]["net_pnl"] == 5.0


class TestGrowingAPosition:
    def test_two_entries_average_their_price_by_quantity(self):
        first = _execution(
            side=OrderSide.BUY,
            quantity=1.0,
            price=100.0,
            fee=1.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        second = _execution(
            side=OrderSide.BUY,
            quantity=1.0,
            price=200.0,
            fee=1.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=None,
        )
        closed = _execution(
            side=OrderSide.SELL,
            quantity=2.0,
            price=180.0,
            fee=2.0,
            timestamp=datetime(2026, 8, 3, tzinfo=UTC),
            realised_profit=60.0,
        )

        trades = build_trades(
            [first, second, closed],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert trades.iloc[0]["entry_price"] == 150.0

    def test_the_entry_reason_stays_the_first_executions(self):
        first = _execution(
            quantity=1.0,
            kind="market",
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        second = _execution(
            quantity=1.0,
            kind="trigger",
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=None,
        )
        closed = _execution(
            side=OrderSide.SELL,
            quantity=2.0,
            timestamp=datetime(2026, 8, 3, tzinfo=UTC),
            realised_profit=5.0,
        )

        trades = build_trades(
            [first, second, closed],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert trades.iloc[0]["entry_reason"] == "market"

    def test_entry_fees_accumulate_across_entries(self):
        first = _execution(
            quantity=1.0,
            fee=1.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        second = _execution(
            quantity=1.0,
            fee=2.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=None,
        )
        closed = _execution(
            side=OrderSide.SELL,
            quantity=2.0,
            fee=0.0,
            timestamp=datetime(2026, 8, 3, tzinfo=UTC),
            realised_profit=10.0,
        )

        trades = build_trades(
            [first, second, closed],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert trades.iloc[0]["entry_fee"] == pytest.approx(3.0)


class TestPartialClose:
    def test_a_partial_close_leaves_the_position_open(self):
        opened = _execution(
            quantity=10.0,
            fee=0.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        partial = _execution(
            side=OrderSide.SELL,
            quantity=4.0,
            fee=0.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=40.0,
        )

        trades = build_trades(
            [opened, partial],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert len(trades) == 1
        assert trades.iloc[0]["net_quantity"] == 4.0

    def test_the_remaining_quantity_keeps_the_original_entry_price(self):
        opened = _execution(
            quantity=10.0,
            price=100.0,
            fee=0.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        partial = _execution(
            side=OrderSide.SELL,
            quantity=4.0,
            price=110.0,
            fee=0.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=40.0,
        )
        rest = _execution(
            side=OrderSide.SELL,
            quantity=6.0,
            price=120.0,
            fee=0.0,
            timestamp=datetime(2026, 8, 3, tzinfo=UTC),
            realised_profit=120.0,
        )

        trades = build_trades(
            [opened, partial, rest],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        second_row = trades[trades["exit_price"] == 120.0].iloc[0]
        assert second_row["entry_price"] == 100.0
        assert second_row["net_quantity"] == 6.0

    def test_entry_fee_is_split_between_the_partial_closes(self):
        opened = _execution(
            quantity=10.0,
            fee=10.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        partial = _execution(
            side=OrderSide.SELL,
            quantity=4.0,
            fee=0.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=40.0,
        )
        rest = _execution(
            side=OrderSide.SELL,
            quantity=6.0,
            fee=0.0,
            timestamp=datetime(2026, 8, 3, tzinfo=UTC),
            realised_profit=120.0,
        )

        trades = build_trades(
            [opened, partial, rest],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert trades["entry_fee"].sum() == pytest.approx(10.0)


class TestFlippingAPosition:
    def _flip(self):
        opened = _execution(
            side=OrderSide.BUY,
            quantity=10.0,
            price=100.0,
            fee=0.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        flip = _execution(
            side=OrderSide.SELL,
            quantity=15.0,
            price=110.0,
            fee=1.5,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=100.0,
        )
        return build_trades(
            [opened, flip],
            since=_SINCE,
            until=_UNTIL,
        ).trades

    def test_the_overflowing_execution_closes_the_original_position(self):
        trades = self._flip()

        assert len(trades) == 1
        assert trades.iloc[0]["side"] == "long"
        assert trades.iloc[0]["net_quantity"] == 10.0
        assert trades.iloc[0]["gross_pnl"] == 100.0

    def test_the_overflow_opens_a_new_position_on_the_other_side(self):
        opened = _execution(
            side=OrderSide.BUY,
            quantity=10.0,
            price=100.0,
            fee=0.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        flip = _execution(
            side=OrderSide.SELL,
            quantity=15.0,
            price=110.0,
            fee=1.5,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=100.0,
        )
        closed = _execution(
            side=OrderSide.BUY,
            quantity=5.0,
            price=90.0,
            fee=0.0,
            timestamp=datetime(2026, 8, 3, tzinfo=UTC),
            realised_profit=20.0,
        )

        trades = build_trades(
            [opened, flip, closed],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        new_leg = trades[trades["exit_price"] == 90.0].iloc[0]
        assert new_leg["side"] == "short"
        assert new_leg["entry_price"] == 110.0
        assert new_leg["net_quantity"] == 5.0

    def test_the_flip_executions_fee_splits_across_both_positions(self):
        trades = self._flip()

        assert trades.iloc[0]["exit_fee"] == pytest.approx(1.0)


class TestUnpairableClose:
    def test_a_close_with_nothing_open_is_not_a_trade(self):
        paired = build_trades(
            [_execution(side=OrderSide.SELL, realised_profit=50.0)],
            since=_SINCE,
            until=_UNTIL,
        )

        assert paired.trades.empty

    def test_a_close_with_nothing_open_is_counted(self):
        paired = build_trades(
            [_execution(side=OrderSide.SELL, realised_profit=50.0)],
            since=_SINCE,
            until=_UNTIL,
        )

        assert paired.unpairable == 1

    def test_a_paired_close_is_not_counted(self):
        opened = _execution(
            quantity=1.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        closed = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=5.0,
        )

        paired = build_trades(
            [opened, closed],
            since=_SINCE,
            until=_UNTIL,
        )

        assert paired.unpairable == 0


class TestMultipleSymbols:
    def test_each_symbol_is_paired_on_its_own(self):
        btc_open = _execution(
            symbol=_BTC,
            quantity=1.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        eth_close = _execution(
            symbol=_ETH, side=OrderSide.SELL, quantity=1.0, realised_profit=5.0
        )
        btc_close = _execution(
            symbol=_BTC,
            side=OrderSide.SELL,
            quantity=1.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=10.0,
        )

        paired = build_trades(
            [btc_open, eth_close, btc_close],
            since=_SINCE,
            until=_UNTIL,
        )

        assert paired.unpairable == 1
        assert paired.trades["symbol"].tolist() == ["BTC/USDT:USDT"]


class TestOrdering:
    def test_trades_are_ordered_by_entry_time_descending(self):
        earlier_open = _execution(
            quantity=1.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        earlier_close = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=5.0,
        )
        later_open = _execution(
            quantity=1.0,
            timestamp=datetime(2026, 8, 3, tzinfo=UTC),
            realised_profit=None,
        )
        later_close = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            timestamp=datetime(2026, 8, 4, tzinfo=UTC),
            realised_profit=8.0,
        )

        trades = build_trades(
            [earlier_open, earlier_close, later_open, later_close],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert trades["entry_time"].tolist() == [
            datetime(2026, 8, 3, tzinfo=UTC),
            datetime(2026, 8, 1, tzinfo=UTC),
        ]


class TestFloatingPointQuantityDrift:
    def test_a_close_within_floating_point_tolerance_of_the_leg_closes_it_fully(self):
        opened = _execution(
            quantity=0.1 + 0.2,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        closed = _execution(
            side=OrderSide.SELL,
            quantity=0.3,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=5.0,
        )

        trades = build_trades(
            [opened, closed],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert len(trades) == 1
        assert trades.iloc[0]["net_quantity"] == pytest.approx(0.3)

    def test_a_leg_within_floating_point_tolerance_of_the_close_leaves_nothing_open(
        self,
    ):
        opened = _execution(
            quantity=0.3,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        closed = _execution(
            side=OrderSide.SELL,
            quantity=0.1 + 0.2,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=5.0,
        )
        reopened = _execution(
            quantity=1.0,
            timestamp=datetime(2026, 8, 3, tzinfo=UTC),
            realised_profit=None,
        )

        trades = build_trades(
            [opened, closed, reopened],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert len(trades) == 1


class TestReconciliationAgainstTheVenue:
    @pytest.fixture
    def executions(self) -> list[Execution]:
        long_open = _execution(
            symbol=_BTC,
            side=OrderSide.BUY,
            quantity=1.0,
            price=100.0,
            fee=1.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        long_close = _execution(
            symbol=_BTC,
            side=OrderSide.SELL,
            quantity=1.0,
            price=110.0,
            fee=1.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=10.0,
        )
        short_open = _execution(
            symbol=_ETH,
            side=OrderSide.SELL,
            quantity=2.0,
            price=50.0,
            fee=2.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        short_close = _execution(
            symbol=_ETH,
            side=OrderSide.BUY,
            quantity=2.0,
            price=40.0,
            fee=2.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=20.0,
        )
        flip_open = _execution(
            symbol=_SOL,
            side=OrderSide.BUY,
            quantity=1.0,
            price=200.0,
            fee=1.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        flip = _execution(
            symbol=_SOL,
            side=OrderSide.SELL,
            quantity=3.0,
            price=210.0,
            fee=3.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=10.0,
        )
        flip_close = _execution(
            symbol=_SOL,
            side=OrderSide.BUY,
            quantity=2.0,
            price=190.0,
            fee=2.0,
            timestamp=datetime(2026, 8, 3, tzinfo=UTC),
            realised_profit=40.0,
        )
        return [
            long_open,
            long_close,
            short_open,
            short_close,
            flip_open,
            flip,
            flip_close,
        ]

    def test_nothing_in_the_fixture_is_unpairable(self, executions):
        paired = build_trades(executions, since=_SINCE, until=_UNTIL)

        assert paired.unpairable == 0

    def test_gross_pnl_sums_to_the_venues_realised_profit(self, executions):
        paired = build_trades(executions, since=_SINCE, until=_UNTIL)

        booked = sum(execution.realised_profit or 0.0 for execution in executions)
        assert paired.trades["gross_pnl"].sum() == pytest.approx(booked)

    def test_every_paired_fee_is_attributed_exactly_once(self, executions):
        paired = build_trades(executions, since=_SINCE, until=_UNTIL)

        booked_fees = sum(execution.fee or 0.0 for execution in executions)
        attributed_fees = (
            paired.trades["entry_fee"].sum() + paired.trades["exit_fee"].sum()
        )
        assert attributed_fees == pytest.approx(booked_fees)

    def test_net_pnl_sums_to_gross_pnl_less_the_attributed_fees(self, executions):
        paired = build_trades(executions, since=_SINCE, until=_UNTIL)

        trades = paired.trades
        attributed_fees = trades["entry_fee"].sum() + trades["exit_fee"].sum()
        assert trades["net_pnl"].sum() == pytest.approx(
            trades["gross_pnl"].sum() - attributed_fees
        )


class TestReadingBackForTheEntry:
    def test_a_close_whose_entry_predates_the_window_uses_the_true_entry(self):
        entry_before_window = _execution(
            quantity=1.0,
            price=100.0,
            timestamp=datetime(2026, 7, 1, tzinfo=UTC),
            realised_profit=None,
        )
        close_inside_window = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            price=110.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=10.0,
        )
        window_start = datetime(2026, 8, 1, tzinfo=UTC)

        trades = build_trades(
            [entry_before_window, close_inside_window],
            since=window_start,
            until=_UNTIL,
        ).trades

        assert len(trades) == 1
        assert trades.iloc[0]["entry_time"] == datetime(2026, 7, 1, tzinfo=UTC)
        assert trades.iloc[0]["entry_price"] == 100.0

    def test_that_trade_is_not_counted_unpairable(self):
        entry_before_window = _execution(
            quantity=1.0,
            timestamp=datetime(2026, 7, 1, tzinfo=UTC),
            realised_profit=None,
        )
        close_inside_window = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=10.0,
        )

        paired = build_trades(
            [entry_before_window, close_inside_window],
            since=datetime(2026, 8, 1, tzinfo=UTC),
            until=_UNTIL,
        )

        assert paired.unpairable == 0

    def test_the_gross_pnl_sum_matches_the_venues_total_for_the_window(self):
        entry_before_window = _execution(
            quantity=1.0,
            timestamp=datetime(2026, 7, 1, tzinfo=UTC),
            realised_profit=None,
        )
        close_before_window = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            timestamp=datetime(2026, 7, 15, tzinfo=UTC),
            realised_profit=999.0,
        )
        reopened = _execution(
            quantity=1.0,
            timestamp=datetime(2026, 7, 20, tzinfo=UTC),
            realised_profit=None,
        )
        close_inside_window = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=10.0,
        )
        window_start = datetime(2026, 8, 1, tzinfo=UTC)
        executions = [
            entry_before_window,
            close_before_window,
            reopened,
            close_inside_window,
        ]

        paired = build_trades(
            executions,
            since=window_start,
            until=_UNTIL,
        )

        venues_total_for_the_window = sum(
            execution.realised_profit or 0.0
            for execution in executions
            if execution.timestamp >= window_start
        )
        assert paired.trades["gross_pnl"].sum() == pytest.approx(
            venues_total_for_the_window
        )

    def test_an_opening_execution_before_the_window_is_not_a_trade_on_its_own(self):
        entry_before_window = _execution(
            quantity=1.0,
            timestamp=datetime(2026, 7, 1, tzinfo=UTC),
            realised_profit=None,
        )

        trades = build_trades(
            [entry_before_window],
            since=datetime(2026, 8, 1, tzinfo=UTC),
            until=_UNTIL,
        ).trades

        assert trades.empty

    def test_a_close_before_the_window_is_dropped_even_when_paired(self):
        entry = _execution(
            quantity=1.0,
            timestamp=datetime(2026, 7, 1, tzinfo=UTC),
            realised_profit=None,
        )
        close_before_window = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            timestamp=datetime(2026, 7, 15, tzinfo=UTC),
            realised_profit=5.0,
        )

        trades = build_trades(
            [entry, close_before_window],
            since=datetime(2026, 8, 1, tzinfo=UTC),
            until=_UNTIL,
        ).trades

        assert trades.empty

    def test_a_close_whose_entry_predates_even_the_history_given_stays_unpairable(
        self,
    ):
        close_inside_window = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=10.0,
        )

        paired = build_trades(
            [close_inside_window],
            since=datetime(2026, 8, 1, tzinfo=UTC),
            until=_UNTIL,
        )

        assert paired.unpairable == 1
        assert paired.trades.empty


class TestTheSideCarriedDecidesOpenFromClose:
    def test_a_break_even_close_closes_the_position(self):
        opened = _execution(side=OrderSide.BUY, quantity=1.0, price=100.0)
        broke_even = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            price=100.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=0.0,
        )

        trades = build_trades(
            [opened, broke_even],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert len(trades) == 1

    def test_a_break_even_close_carries_no_gross_profit(self):
        opened = _execution(side=OrderSide.BUY, quantity=1.0, price=100.0)
        broke_even = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            price=100.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=0.0,
        )

        trades = build_trades(
            [opened, broke_even],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert trades.iloc[0]["gross_pnl"] == 0.0

    def test_a_break_even_close_leaves_nothing_open(self):
        opened = _execution(side=OrderSide.BUY, quantity=1.0, price=100.0)
        broke_even = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            price=100.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=0.0,
        )
        reopened = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            price=90.0,
            timestamp=datetime(2026, 8, 3, tzinfo=UTC),
        )
        bought_back = _execution(
            side=OrderSide.BUY,
            quantity=1.0,
            price=80.0,
            timestamp=datetime(2026, 8, 4, tzinfo=UTC),
            realised_profit=10.0,
        )

        trades = build_trades(
            [opened, broke_even, reopened, bought_back],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert list(trades["side"]) == ["short", "long"]

    def test_an_execution_on_the_side_already_held_grows_the_position(self):
        opened = _execution(side=OrderSide.BUY, quantity=1.0, price=100.0)
        added = _execution(
            side=OrderSide.BUY,
            quantity=1.0,
            price=200.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
        )
        closed = _execution(
            side=OrderSide.SELL,
            quantity=2.0,
            price=160.0,
            timestamp=datetime(2026, 8, 3, tzinfo=UTC),
            realised_profit=20.0,
        )

        trades = build_trades(
            [opened, added, closed],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert trades.iloc[0]["entry_price"] == pytest.approx(150.0)


class TestTheWindowEnd:
    def test_a_close_after_the_window_is_not_a_trade(self):
        opened = _execution(
            side=OrderSide.BUY, timestamp=datetime(2026, 8, 1, tzinfo=UTC)
        )
        closed_later = _execution(
            side=OrderSide.SELL,
            timestamp=datetime(2026, 9, 1, tzinfo=UTC),
            realised_profit=10.0,
        )

        trades = build_trades(
            [opened, closed_later],
            since=_SINCE,
            until=datetime(2026, 8, 15, tzinfo=UTC),
        ).trades

        assert trades.empty

    def test_a_close_after_the_window_is_not_counted_unpairable(self):
        closed_later = _execution(
            side=OrderSide.SELL,
            timestamp=datetime(2026, 9, 1, tzinfo=UTC),
            realised_profit=10.0,
        )

        paired = build_trades(
            [closed_later],
            since=_SINCE,
            until=datetime(2026, 8, 15, tzinfo=UTC),
        )

        assert paired.unpairable == 0

    def test_a_close_exactly_at_the_window_end_is_a_trade(self):
        end = datetime(2026, 8, 15, tzinfo=UTC)
        opened = _execution(
            side=OrderSide.BUY, timestamp=datetime(2026, 8, 1, tzinfo=UTC)
        )
        closed = _execution(side=OrderSide.SELL, timestamp=end, realised_profit=10.0)

        trades = build_trades([opened, closed], since=_SINCE, until=end).trades

        assert len(trades) == 1


class TestTagAttribution:
    def test_the_tag_comes_from_the_closing_execution(self):
        opened = _execution(
            client_order_id=client_order_id_carrying("opened"),
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        closed = _execution(
            client_order_id=client_order_id_carrying("closed"),
            side=OrderSide.SELL,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=5.0,
        )

        trades = build_trades([opened, closed], since=_SINCE, until=_UNTIL).trades

        assert trades.iloc[0]["tag"] == "closed"

    def test_partial_closes_on_a_stacked_symbol_each_carry_their_own_closing_tag(
        self,
    ):
        hour_open = _execution(
            order_id="hour-open",
            price=100.0,
            timestamp=datetime(2026, 8, 1, tzinfo=UTC),
            realised_profit=None,
        )
        two_hour_open = _execution(
            order_id="two-hour-open",
            price=200.0,
            timestamp=datetime(2026, 8, 1, 0, 0, 1, tzinfo=UTC),
            realised_profit=None,
        )
        hour_close = _execution(
            order_id="hour-close",
            side=OrderSide.SELL,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=10.0,
            fee=1.0,
            client_order_id="abc123def456-1-1h-alpha",
        )
        two_hour_close = _execution(
            order_id="two-hour-close",
            side=OrderSide.SELL,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=20.0,
            fee=1.0,
            client_order_id="abc123def456-2-2h-alpha",
        )

        trades = build_trades(
            [hour_open, two_hour_open, hour_close, two_hour_close],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert set(trades["tag"]) == {"1h-alpha", "2h-alpha"}


class TestACloseTheVenueStatesNoProfitFor:
    def _closed(self, profit):
        opened = _execution(side=OrderSide.BUY, quantity=1.0, price=100.0)
        closed = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            price=125.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=profit,
        )
        return build_trades(
            [opened, closed],
            since=_SINCE,
            until=_UNTIL,
        )

    def test_it_yields_no_trade(self):
        assert self._closed(None).trades.empty

    def test_it_is_counted(self):
        assert self._closed(None).unpriced == 1

    def test_a_stated_profit_of_zero_still_yields_a_trade(self):
        paired = self._closed(0.0)

        assert len(paired.trades) == 1

    def test_a_stated_profit_of_zero_is_not_counted_unpriced(self):
        assert self._closed(0.0).unpriced == 0

    def test_a_stated_profit_is_carried_whole(self):
        assert self._closed(25.0).trades.iloc[0]["gross_pnl"] == 25.0

    def test_the_position_it_closed_is_still_closed(self):
        opened = _execution(side=OrderSide.BUY, quantity=1.0, price=100.0)
        unpriced_close = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            price=125.0,
            timestamp=datetime(2026, 8, 2, tzinfo=UTC),
            realised_profit=None,
        )
        reopened = _execution(
            side=OrderSide.BUY,
            quantity=1.0,
            price=130.0,
            timestamp=datetime(2026, 8, 3, tzinfo=UTC),
        )
        closed = _execution(
            side=OrderSide.SELL,
            quantity=1.0,
            price=140.0,
            timestamp=datetime(2026, 8, 4, tzinfo=UTC),
            realised_profit=10.0,
        )

        trades = build_trades(
            [opened, unpriced_close, reopened, closed],
            since=_SINCE,
            until=_UNTIL,
        ).trades

        assert trades.iloc[0]["entry_price"] == 130.0
