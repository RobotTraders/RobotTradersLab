from collections.abc import Callable
from datetime import datetime
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab._core.account_snapshot import NamedAccount
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.exchanges import (
    Balance,
    Execution,
    FillEffect,
    MarginMode,
    MarginSettings,
    OrderProtocol,
    OrderSide,
    OrderType,
    PositionSnapshot,
)
from robottraderslab.strategies import AccountSnapshot, AccountSnapshots

ACCOUNT_NAME = "test"
BTCUSDT = Symbol.create("BTC/USDT:USDT")
ETHUSDT = Symbol.create("ETH/USDT:USDT")


@pytest.fixture
def create_execution() -> Callable[[Symbol], Execution]:
    def _create_execution(
        symbol: Symbol,
        *,
        kind: OrderType | None = None,
        execution_id: str = "execution-1",
        effect: FillEffect | None = None,
    ) -> Execution:
        return Execution(
            execution_id=execution_id,
            order_id="order-1",
            symbol=symbol,
            side=OrderSide.BUY,
            price=100.0,
            quantity=1.0,
            timestamp=datetime(2026, 5, 1),
            kind=kind,
            effect=effect,
        )

    return _create_execution


def _order_on(symbol: Symbol, kind: OrderType = "limit") -> Mock:
    order = Mock(spec=OrderProtocol)
    order.symbol = symbol
    order.kind = kind
    return order


def _account_named(name: str) -> Mock:
    account = Mock(spec=NamedAccount)
    account.name = name
    return account


class TestAccountSnapshotAccessors:
    def test_stop_loss_orders_among_other_kinds_and_symbols(self):
        rung_stop_loss = _order_on(BTCUSDT, "stop-loss")
        position_stop_loss = _order_on(BTCUSDT, "stop-loss")
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME,
            open_orders=[
                _order_on(BTCUSDT),
                rung_stop_loss,
                _order_on(ETHUSDT, "stop-loss"),
                position_stop_loss,
            ],
        )

        assert snapshot.stop_loss_orders(BTCUSDT) == [
            rung_stop_loss,
            position_stop_loss,
        ]

    def test_take_profit_orders_alongside_a_resting_stop_loss(self):
        take_profit = _order_on(BTCUSDT, "take-profit")
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME,
            open_orders=[_order_on(BTCUSDT, "stop-loss"), take_profit],
        )

        assert snapshot.take_profit_orders(BTCUSDT) == [take_profit]

    def test_take_profit_orders_when_none_rests(self):
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME, open_orders=[_order_on(BTCUSDT, "stop-loss")]
        )

        assert snapshot.take_profit_orders(BTCUSDT) == []

    def test_open_orders_filtered_by_symbol(self):
        btc_order = _order_on(BTCUSDT)
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME, open_orders=[btc_order, _order_on(ETHUSDT)]
        )

        assert snapshot.open_orders(BTCUSDT) == [btc_order]

    def test_position_returns_none_when_flat(self):
        snapshot = AccountSnapshot(account_name=ACCOUNT_NAME, positions={})

        assert snapshot.position(BTCUSDT) is None

    def test_position_returned_for_symbol(self):
        position = Mock(spec=PositionSnapshot)
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME, positions={BTCUSDT: position}
        )

        assert snapshot.position(BTCUSDT) is position

    def test_balance_returned_for_currency(self):
        balance = Balance.compute(available=800.0, locked=200.0)
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME, balances={"USDT": balance}
        )

        assert snapshot.balance("USDT") == balance

    def test_equity_returned_for_currency(self):
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME, equities={"USDT": 10_000.0}
        )

        assert snapshot.equity("USDT") == 10_000.0

    def test_margin_settings_returned_for_symbol(self):
        settings = MarginSettings(leverage=5.0, margin_mode=MarginMode.ISOLATED)
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME, margin_settings={BTCUSDT: settings}
        )

        assert snapshot.margin_settings(BTCUSDT) == settings

    def test_stop_loss_fills_filtered_by_symbol(self, create_execution):
        btc_fill = create_execution(BTCUSDT, kind="stop-loss")
        eth_fill = create_execution(ETHUSDT, kind="stop-loss")
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME,
            executions=[btc_fill, eth_fill],
            executions_declared=True,
        )

        assert snapshot.stop_loss_fills(BTCUSDT) == [btc_fill]

    def test_take_profit_fills_filtered_by_symbol(self, create_execution):
        btc_fill = create_execution(BTCUSDT, kind="take-profit")
        eth_fill = create_execution(ETHUSDT, kind="take-profit")
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME,
            executions=[btc_fill, eth_fill],
            executions_declared=True,
        )

        assert snapshot.take_profit_fills(BTCUSDT) == [btc_fill]

    def test_liquidation_fills_filtered_by_symbol(self, create_execution):
        btc_fill = create_execution(BTCUSDT, kind="liquidation")
        eth_fill = create_execution(ETHUSDT, kind="liquidation")
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME,
            executions=[btc_fill, eth_fill],
            executions_declared=True,
        )

        assert snapshot.liquidation_fills(BTCUSDT) == [btc_fill]

    def test_entry_fills_filtered_by_symbol(self, create_execution):
        btc_fill = create_execution(BTCUSDT, kind="trigger")
        eth_fill = create_execution(ETHUSDT, kind="trigger")
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME,
            executions=[btc_fill, eth_fill],
            executions_declared=True,
        )

        assert snapshot.entry_fills(BTCUSDT) == [btc_fill]

    def test_entry_fills_include_limit_and_trigger_kinds(self, create_execution):
        limit_fill = create_execution(BTCUSDT, kind="limit", execution_id="limit-1")
        trigger_fill = create_execution(
            BTCUSDT, kind="trigger", execution_id="trigger-1"
        )
        stop_loss_fill = create_execution(
            BTCUSDT, kind="stop-loss", execution_id="sl-1"
        )
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME,
            executions=[limit_fill, trigger_fill, stop_loss_fill],
            executions_declared=True,
        )

        assert snapshot.entry_fills(BTCUSDT) == [limit_fill, trigger_fill]

    def test_execution_without_a_kind_is_excluded_from_every_typed_accessor(
        self, create_execution
    ):
        kindless = create_execution(BTCUSDT)
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME, executions=[kindless], executions_declared=True
        )

        assert snapshot.entry_fills(BTCUSDT) == []
        assert snapshot.stop_loss_fills(BTCUSDT) == []
        assert snapshot.take_profit_fills(BTCUSDT) == []
        assert snapshot.liquidation_fills(BTCUSDT) == []


class TestUndeclaredAccess:
    def test_open_orders_names_the_missing_declaration(self):
        snapshot = AccountSnapshot(account_name=ACCOUNT_NAME)

        with pytest.raises(StrategyCriticalError, match="open_orders=True"):
            snapshot.open_orders(BTCUSDT)

    def test_stop_loss_orders_names_the_missing_declaration(self):
        snapshot = AccountSnapshot(account_name=ACCOUNT_NAME)

        with pytest.raises(StrategyCriticalError, match="open_orders=True"):
            snapshot.stop_loss_orders(BTCUSDT)

    def test_position_names_the_missing_declaration(self):
        snapshot = AccountSnapshot(account_name=ACCOUNT_NAME)

        with pytest.raises(StrategyCriticalError, match="positions=True"):
            snapshot.position(BTCUSDT)

    def test_balance_names_the_missing_declaration(self):
        snapshot = AccountSnapshot(account_name=ACCOUNT_NAME)

        with pytest.raises(StrategyCriticalError, match="balances=True"):
            snapshot.balance("USDT")

    def test_balance_in_a_currency_no_declared_symbol_carries(self):
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME,
            balances={"USDT": Balance.compute(available=800.0, locked=200.0)},
        )

        with pytest.raises(
            StrategyCriticalError, match=r"EUR.*symbols=\[\.\.\.\], balances=True"
        ):
            snapshot.balance("EUR")

    def test_equity_names_the_missing_declaration(self):
        snapshot = AccountSnapshot(account_name=ACCOUNT_NAME)

        with pytest.raises(StrategyCriticalError, match="equity"):
            snapshot.equity("USDT")

    def test_equity_raises_for_undeclared_currency(self):
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME, equities={"USDT": 10_000.0}
        )

        with pytest.raises(StrategyCriticalError, match="USDC"):
            snapshot.equity("USDC")

    def test_margin_settings_names_the_missing_declaration(self):
        snapshot = AccountSnapshot(account_name=ACCOUNT_NAME)

        with pytest.raises(StrategyCriticalError, match="margin_settings"):
            snapshot.margin_settings(BTCUSDT)

    def test_margin_settings_raises_for_undeclared_symbol(self):
        settings = MarginSettings(leverage=5.0, margin_mode=MarginMode.ISOLATED)
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME, margin_settings={BTCUSDT: settings}
        )

        with pytest.raises(StrategyCriticalError, match="ETH"):
            snapshot.margin_settings(ETHUSDT)

    def test_stop_loss_fills_name_the_missing_declaration(self):
        snapshot = AccountSnapshot(account_name=ACCOUNT_NAME)

        with pytest.raises(StrategyCriticalError, match="executions=True"):
            snapshot.stop_loss_fills(BTCUSDT)

    def test_stop_loss_fills_raise_even_when_fetched_for_reporting(
        self, create_execution
    ):
        fetched_for_reporting = create_execution(BTCUSDT, kind="stop-loss")
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME, executions=[fetched_for_reporting]
        )

        with pytest.raises(StrategyCriticalError, match="executions=True"):
            snapshot.stop_loss_fills(BTCUSDT)

    def test_error_names_the_account(self):
        snapshot = AccountSnapshot(account_name="hedge")

        with pytest.raises(StrategyCriticalError, match="hedge"):
            snapshot.position(BTCUSDT)


class TestDeclaredExecutions:
    def test_returns_every_kind_for_the_symbol(self, create_execution):
        stop_loss = create_execution(BTCUSDT, kind="stop-loss")
        kindless = create_execution(BTCUSDT, execution_id="execution-2")
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME,
            executions=[stop_loss, kindless],
            executions_declared=True,
        )

        assert snapshot._declared_executions(BTCUSDT) == [stop_loss, kindless]

    def test_filtered_by_symbol(self, create_execution):
        btc_fill = create_execution(BTCUSDT)
        eth_fill = create_execution(ETHUSDT)
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME,
            executions=[btc_fill, eth_fill],
            executions_declared=True,
        )

        assert snapshot._declared_executions(BTCUSDT) == [btc_fill]

    def test_returns_fills_even_when_not_declared(self, create_execution):
        fetched_for_reporting = create_execution(BTCUSDT)
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME, executions=[fetched_for_reporting]
        )

        assert snapshot._declared_executions(BTCUSDT) == [fetched_for_reporting]


class TestVenueFiredFills:
    def test_returns_every_venue_fired_kind_undeclared(self, create_execution):
        stop_loss = create_execution(BTCUSDT, kind="stop-loss")
        take_profit = create_execution(BTCUSDT, kind="take-profit")
        liquidation = create_execution(BTCUSDT, kind="liquidation")
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME,
            executions=[stop_loss, take_profit, liquidation],
        )

        fills = snapshot._venue_fired_fills(BTCUSDT)

        assert fills == [stop_loss, take_profit, liquidation]

    def test_filtered_by_symbol(self, create_execution):
        btc_fill = create_execution(BTCUSDT, kind="stop-loss")
        eth_fill = create_execution(ETHUSDT, kind="stop-loss")
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME, executions=[btc_fill, eth_fill]
        )

        assert snapshot._venue_fired_fills(BTCUSDT) == [btc_fill]

    def test_excludes_entry_triggers_and_kindless_executions(self, create_execution):
        trigger = create_execution(BTCUSDT, kind="trigger")
        kindless = create_execution(BTCUSDT)
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME, executions=[trigger, kindless]
        )

        assert snapshot._venue_fired_fills(BTCUSDT) == []

    def test_returns_nothing_when_no_executions_were_fetched(self):
        snapshot = AccountSnapshot(account_name=ACCOUNT_NAME)

        assert snapshot._venue_fired_fills(BTCUSDT) == []


class TestAttributedEffects:
    def test_keyed_by_execution(self, create_execution):
        attributed = create_execution(BTCUSDT, execution_id="e1", effect="open")
        snapshot = AccountSnapshot(account_name=ACCOUNT_NAME, executions=[attributed])

        assert snapshot._effects_by_execution_id() == {"e1": "open"}

    def test_unattributed_executions_are_left_out(self, create_execution):
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME,
            executions=[create_execution(BTCUSDT, execution_id="e1")],
        )

        assert snapshot._effects_by_execution_id() == {}

    def test_account_with_no_snapshot_attributes_nothing(self):
        account_snapshots = AccountSnapshots({})

        assert account_snapshots._effects_by_execution_id(_account_named("hedge")) == {}

    def test_account_with_a_snapshot_reads_its_own(self, create_execution):
        attributed = create_execution(BTCUSDT, execution_id="e1", effect="close")
        snapshot = AccountSnapshot(account_name="main", executions=[attributed])

        account_snapshots = AccountSnapshots({"main": snapshot})

        assert account_snapshots._effects_by_execution_id(_account_named("main")) == {
            "e1": "close"
        }


class TestAccountSnapshots:
    def test_snapshot_returned_for_its_account(self):
        account = _account_named("main")
        snapshot = AccountSnapshot(account_name="main", positions={})

        account_snapshots = AccountSnapshots({"main": snapshot})

        assert account_snapshots.of(account) is snapshot

    def test_undeclared_account_names_itself(self):
        account_snapshots = AccountSnapshots({})

        with pytest.raises(StrategyCriticalError, match="hedge"):
            account_snapshots.of(_account_named("hedge"))
