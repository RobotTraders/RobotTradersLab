import logging
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab._core import sweep_orphan_protections
from robottraderslab.exchanges import PositionSide
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    BookKeeper,
    OrderProtocol,
    StrategyRequirements,
)
from robottraderslab.strategies.futures import (
    CancelOrderByIdAction,
    PositionSnapshot,
)

ACCOUNT_NAME = "test"
BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")


class _Account:
    def __init__(self) -> None:
        self.name = ACCOUNT_NAME
        self.cancelled: list[tuple[Symbol, str]] = []

    def cancel_order(self, symbol: Symbol, order_id: str) -> CancelOrderByIdAction:
        self.cancelled.append((symbol, order_id))
        return CancelOrderByIdAction(exchange=Mock(), symbol=symbol, order_id=order_id)


def _order(kind: str, order_id: str = "sl-1", symbol: Symbol = BTC) -> Mock:
    order = Mock(spec=OrderProtocol)
    order.kind = kind
    order.order_id = order_id
    order.symbol = symbol
    return order


def _position(symbol: Symbol = BTC) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        side=PositionSide.LONG,
        quantity=1.0,
        average_entry_price=100.0,
        entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        leverage=1.0,
        liquidation_price=0.0,
    )


def _snapshots(
    *,
    positions: dict[Symbol, PositionSnapshot] | None = None,
    open_orders: list[Mock] | None = None,
) -> AccountSnapshots:
    return AccountSnapshots(
        {
            ACCOUNT_NAME: AccountSnapshot(
                account_name=ACCOUNT_NAME,
                positions=positions or {},
                open_orders=open_orders or [],
            )
        }
    )


@pytest.fixture
def account() -> _Account:
    return _Account()


def _declared(account: _Account, **overrides) -> StrategyRequirements:
    requirements = StrategyRequirements()
    requirements.account.add(
        account,
        symbols=overrides.pop("symbols", [BTC]),
        positions=overrides.pop("positions", True),
        open_orders=overrides.pop("open_orders", True),
        **overrides,
    )
    return requirements


class TestSweepOrphanProtections:
    def test_cancels_a_stop_loss_left_on_a_flat_symbol(self, account):
        requirements = _declared(account)
        snapshots = _snapshots(open_orders=[_order("stop-loss")])

        sweep_orphan_protections(requirements.account, snapshots, BookKeeper())

        assert account.cancelled == [(BTC, "sl-1")]

    def test_cancels_a_take_profit_left_on_a_flat_symbol(self, account):
        requirements = _declared(account)
        snapshots = _snapshots(open_orders=[_order("take-profit", "tp-1")])

        sweep_orphan_protections(requirements.account, snapshots, BookKeeper())

        assert account.cancelled == [(BTC, "tp-1")]

    def test_books_the_cancel(self, account):
        requirements = _declared(account)
        snapshots = _snapshots(open_orders=[_order("stop-loss")])
        bookkeeper = BookKeeper()

        sweep_orphan_protections(requirements.account, snapshots, bookkeeper)

        assert len(bookkeeper.list_actions()) == 1

    def test_a_protection_guarding_an_open_position_stays(self, account):
        requirements = _declared(account)
        snapshots = _snapshots(
            positions={BTC: _position()}, open_orders=[_order("stop-loss")]
        )

        sweep_orphan_protections(requirements.account, snapshots, BookKeeper())

        assert account.cancelled == []

    @pytest.mark.parametrize("kind", ["limit", "market", "trigger"])
    def test_other_kinds_are_left_alone(self, account, kind):
        requirements = _declared(account)
        snapshots = _snapshots(open_orders=[_order(kind)])

        sweep_orphan_protections(requirements.account, snapshots, BookKeeper())

        assert account.cancelled == []

    def test_an_account_that_opted_out(self, account):
        requirements = _declared(account, sweep_protections=False)
        snapshots = _snapshots(open_orders=[_order("stop-loss")])

        sweep_orphan_protections(requirements.account, snapshots, BookKeeper())

        assert account.cancelled == []

    def test_an_account_that_never_reads_its_orders(self, account):
        requirements = _declared(account, open_orders=False)
        snapshots = _snapshots(open_orders=[_order("stop-loss")])

        sweep_orphan_protections(requirements.account, snapshots, BookKeeper())

        assert account.cancelled == []

    def test_an_account_that_never_reads_its_positions(self, account):
        requirements = _declared(account, positions=False)
        snapshots = _snapshots(open_orders=[_order("stop-loss")])

        sweep_orphan_protections(requirements.account, snapshots, BookKeeper())

        assert account.cancelled == []

    def test_only_the_declared_symbols_are_swept(self, account):
        requirements = _declared(account, symbols=[BTC])
        snapshots = _snapshots(
            open_orders=[
                _order("stop-loss", "btc-sl", BTC),
                _order("stop-loss", "eth-sl", ETH),
            ]
        )

        sweep_orphan_protections(requirements.account, snapshots, BookKeeper())

        assert account.cancelled == [(BTC, "btc-sl")]

    def test_every_orphan_on_a_symbol_goes(self, account):
        requirements = _declared(account)
        snapshots = _snapshots(
            open_orders=[
                _order("stop-loss", "sl-1"),
                _order("take-profit", "tp-1"),
            ]
        )

        sweep_orphan_protections(requirements.account, snapshots, BookKeeper())

        assert account.cancelled == [(BTC, "sl-1"), (BTC, "tp-1")]

    def test_the_cancellation_is_reported(self, account, caplog):
        requirements = _declared(account)
        snapshots = _snapshots(open_orders=[_order("stop-loss")])

        with caplog.at_level(logging.WARNING):
            sweep_orphan_protections(requirements.account, snapshots, BookKeeper())

        assert "BTC/USDT:USDT" in caplog.text
        assert "stop-loss" in caplog.text

    def test_a_flat_symbol_is_swept_while_a_held_one_is_not(self, account):
        requirements = _declared(account, symbols=[BTC, ETH])
        snapshots = _snapshots(
            positions={ETH: _position(ETH)},
            open_orders=[
                _order("stop-loss", "btc-sl", BTC),
                _order("stop-loss", "eth-sl", ETH),
            ],
        )

        sweep_orphan_protections(requirements.account, snapshots, BookKeeper())

        assert account.cancelled == [(BTC, "btc-sl")]

    def test_every_declared_account_is_swept(self):
        first, second = _Account(), _Account()
        second.name = "second"
        requirements = _declared(first)
        requirements.account.add(
            second, symbols=[BTC], positions=True, open_orders=True
        )
        snapshots = AccountSnapshots(
            {
                name: AccountSnapshot(
                    account_name=name,
                    positions={},
                    open_orders=[_order("stop-loss", f"{name}-sl")],
                )
                for name in (ACCOUNT_NAME, "second")
            }
        )

        sweep_orphan_protections(requirements.account, snapshots, BookKeeper())

        assert first.cancelled == [(BTC, f"{ACCOUNT_NAME}-sl")]
        assert second.cancelled == [(BTC, "second-sl")]
