from datetime import datetime, timezone

import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.exchanges import (
    PositionSide,
    client_order_id_carrying,
)
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    Execution,
    OrderSide,
    StrategyRequirements,
    TrackingId,
    profile_tag,
)
from robottraderslab.strategies.futures import PositionSnapshot

ACCOUNT_NAME = "test"
ALPHA = TrackingId("BTC/USDT:USDT@1d#alpha")
BETA = TrackingId("BTC/USDT:USDT@1d#beta")
ALPHA_TAG = profile_tag("1h", "a")
BETA_TAG = profile_tag("1h", "b")
CANDLE_OPEN = datetime(2026, 1, 1, tzinfo=timezone.utc)
CANDLE_CLOSE = datetime(2026, 1, 2, tzinfo=timezone.utc)


class _Account:
    def __init__(self, name: str = ACCOUNT_NAME) -> None:
        self.name = name


@pytest.fixture
def btc() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def account() -> _Account:
    return _Account()


@pytest.fixture
def requirements() -> StrategyRequirements:
    return StrategyRequirements()


def _snapshots(
    btc: Symbol, quantity: float | None, executions: list[Execution] | None = None
) -> AccountSnapshots:
    positions = (
        {}
        if quantity is None
        else {
            btc: PositionSnapshot(
                symbol=btc,
                side=PositionSide.LONG,
                quantity=quantity,
                average_entry_price=100.0,
                entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
                leverage=1.0,
                liquidation_price=None,
            )
        }
    )
    return AccountSnapshots(
        {
            ACCOUNT_NAME: AccountSnapshot(
                account_name=ACCOUNT_NAME,
                positions=positions,
                executions=executions,
                executions_declared=executions is not None,
            )
        }
    )


def _refresh(requirements: StrategyRequirements, snapshots: AccountSnapshots) -> None:
    requirements.tracker.refresh_all(snapshots, CANDLE_OPEN, CANDLE_CLOSE)


class TestDeclaringATracker:
    def test_a_declared_id_holds_nothing_until_the_venue_reports_a_fill(
        self, requirements, account, btc
    ):
        tracker = requirements.tracker.add(
            account, symbols={ALPHA: btc}, tags={ALPHA: ALPHA_TAG}
        )

        assert tracker.get(ALPHA) is None

    def test_an_untagged_id_is_refused(self, requirements, account, btc):
        with pytest.raises(StrategyCriticalError, match="have no tag"):
            requirements.tracker.add(
                account, symbols={ALPHA: btc, BETA: btc}, tags={ALPHA: ALPHA_TAG}
            )

    def test_two_ids_sharing_a_tag_on_one_symbol_are_refused(
        self, requirements, account, btc
    ):
        with pytest.raises(StrategyCriticalError, match="share the tag"):
            requirements.tracker.add(
                account,
                symbols={ALPHA: btc, BETA: btc},
                tags={ALPHA: ALPHA_TAG, BETA: ALPHA_TAG},
            )

    def test_declaring_reads_the_account_positions(self, requirements, account, btc):
        requirements.account.add(account)
        requirements.tracker.add(account, symbols={ALPHA: btc}, tags={ALPHA: ALPHA_TAG})

        declared = requirements.account._get_all()

        assert declared[0].positions is True
        assert btc in declared[0].symbols

    def test_declaring_reads_the_account_executions(self, requirements, account, btc):
        requirements.account.add(account)
        requirements.tracker.add(account, symbols={ALPHA: btc}, tags={ALPHA: ALPHA_TAG})

        assert requirements.account._get_all()[0].executions is True

    def test_declaring_before_the_account_still_reads_positions_and_executions(
        self, requirements, account, btc
    ):
        requirements.tracker.add(account, symbols={ALPHA: btc}, tags={ALPHA: ALPHA_TAG})
        requirements.account.add(account)

        declared = requirements.account._get_all()

        assert declared[0].positions is True
        assert declared[0].executions is True
        assert btc in declared[0].symbols

    def test_a_tracked_symbol_joins_the_account_symbols(
        self, requirements, account, btc
    ):
        eth = Symbol.create("ETH/USDT:USDT")
        requirements.account.add(account, symbols=[btc])
        requirements.tracker.add(account, symbols={ALPHA: eth}, tags={ALPHA: ALPHA_TAG})

        declared = requirements.account._get_all()

        assert declared[0].symbols == (btc, eth)

    def test_a_second_declaration_of_a_symbol_on_the_account_is_refused(
        self, requirements, account, btc
    ):
        requirements.tracker.add(account, symbols={ALPHA: btc}, tags={ALPHA: ALPHA_TAG})

        with pytest.raises(StrategyCriticalError, match="already tracked"):
            requirements.tracker.add(
                account, symbols={BETA: btc}, tags={BETA: BETA_TAG}
            )

    def test_the_same_symbol_on_another_account_is_accepted(
        self, requirements, account, btc
    ):
        requirements.tracker.add(account, symbols={ALPHA: btc}, tags={ALPHA: ALPHA_TAG})

        tracker = requirements.tracker.add(
            _Account("second"), symbols={BETA: btc}, tags={BETA: BETA_TAG}
        )

        assert tracker.get(BETA) is None


class TestRefreshingDeclaredTrackers:
    def test_refresh_all_folds_the_declared_tags(self, requirements, account, btc):
        tracker = requirements.tracker.add(
            account, symbols={ALPHA: btc}, tags={ALPHA: ALPHA_TAG}
        )
        opened = Execution(
            execution_id="e1",
            order_id="o1",
            symbol=btc,
            side=OrderSide.BUY,
            price=100.0,
            quantity=4.0,
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            effect="open",
            client_order_id=client_order_id_carrying("1h-a"),
        )

        _refresh(requirements, _snapshots(btc, 4.0, executions=[opened]))

        assert tracker.get(ALPHA).quantity == 4.0

    def test_every_declared_tracker_is_refreshed(self, requirements, account, btc):
        first = requirements.tracker.add(
            account, symbols={ALPHA: btc}, tags={ALPHA: ALPHA_TAG}
        )
        second = requirements.tracker.add(
            _Account("second"), symbols={BETA: btc}, tags={BETA: BETA_TAG}
        )
        opened = Execution(
            execution_id="e1",
            order_id="o1",
            symbol=btc,
            side=OrderSide.BUY,
            price=100.0,
            quantity=4.0,
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            effect="open",
            client_order_id=client_order_id_carrying("1h-a"),
        )
        snapshots = AccountSnapshots(
            {
                ACCOUNT_NAME: _snapshots(btc, 4.0, executions=[opened]).of(account),
                "second": AccountSnapshot(
                    account_name="second",
                    positions={},
                    executions=[],
                    executions_declared=True,
                ),
            }
        )

        _refresh(requirements, snapshots)

        assert first.get(ALPHA).quantity == 4.0
        assert second.get(BETA) is None
