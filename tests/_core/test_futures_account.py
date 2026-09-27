from datetime import datetime
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab._core import AccountRequirement
from robottraderslab.exceptions import (
    ExchangeRecoverableError,
    ExchangeTransientError,
    StrategyCriticalError,
)
from robottraderslab.exchanges import (
    Balance,
    Execution,
    FuturesExchangeProtocol,
    MarginMode,
    OrderSide,
    OrderType,
    PositionSide,
    PositionSnapshot,
)
from robottraderslab.futures import FuturesAccount
from robottraderslab.futures.futures_cancel_order import CancelOrderByIdAction
from robottraderslab.futures.futures_cancel_orders import (
    CancelOrdersForSymbolAction,
)
from robottraderslab.futures.futures_set_leverage import SetLeverageAction
from robottraderslab.futures.futures_set_margin_mode import SetMarginModeAction
from robottraderslab.futures.futures_update_position_stop_loss import (
    UpdatePositionStopLossAction,
)
from robottraderslab.futures.futures_update_position_take_profit import (
    UpdatePositionTakeProfitAction,
)
from robottraderslab.strategies import TrackedPosition

BTCUSDT = Symbol.create("BTC/USDT:USDT")
WINDOW_START = datetime(2026, 5, 1)


def _execution(
    execution_id: str,
    side: OrderSide,
    quantity: float,
    minute: int,
    kind: OrderType = "market",
) -> Execution:
    return Execution(
        execution_id=execution_id,
        order_id=execution_id,
        symbol=BTCUSDT,
        side=side,
        price=100.0,
        quantity=quantity,
        timestamp=WINDOW_START.replace(minute=minute),
        kind=kind,
    )


def _long(quantity: float) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=BTCUSDT,
        side=PositionSide.LONG,
        quantity=quantity,
        average_entry_price=100.0,
        entry_time=WINDOW_START,
        leverage=1.0,
        liquidation_price=1.0,
    )


@pytest.fixture
def exchange() -> Mock:
    exchange = Mock(spec=FuturesExchangeProtocol)
    exchange.get_open_positions.return_value = {}
    return exchange


def test_set_margin_mode(exchange):
    account = FuturesAccount(exchange)

    action = account.set_margin_mode(BTCUSDT, MarginMode.CROSS)

    assert isinstance(action, SetMarginModeAction)
    assert action.exchange is exchange
    assert action.symbol == BTCUSDT
    assert action.margin_mode == MarginMode.CROSS


def test_set_leverage(exchange):
    account = FuturesAccount(exchange)

    action = account.set_leverage(BTCUSDT, 1.2)

    assert isinstance(action, SetLeverageAction)
    assert action.exchange is exchange
    assert action.symbol == BTCUSDT
    assert action.leverage == 1.2


@pytest.mark.parametrize("leverage", [0.5, float("nan")])
def test_set_leverage_below_one(exchange, leverage):
    account = FuturesAccount(exchange)

    with pytest.raises(ValueError, match="`leverage` must be at least 1"):
        account.set_leverage(BTCUSDT, leverage)


def test_cancel_order(exchange):
    account = FuturesAccount(exchange)

    action = account.cancel_order(BTCUSDT, "order-123")

    assert isinstance(action, CancelOrderByIdAction)
    assert action.exchange is exchange
    assert action.symbol == BTCUSDT
    assert action.order_id == "order-123"


def test_cancel_orders(exchange):
    account = FuturesAccount(exchange)

    action = account.cancel_orders(BTCUSDT)

    assert isinstance(action, CancelOrdersForSymbolAction)
    assert action.exchange is exchange
    assert action.symbol == BTCUSDT


def test_cancel_by_a_blank_tag(exchange):
    account = FuturesAccount(exchange)

    with pytest.raises(ValueError, match="Tag cannot be empty"):
        account.cancel_orders(BTCUSDT, tag=" ")


def test_move_stop_loss(exchange):
    account = FuturesAccount(exchange)

    action = account.move_stop_loss(BTCUSDT, 42_000.0)

    assert isinstance(action, UpdatePositionStopLossAction)
    assert action.exchange is exchange
    assert action.symbol == BTCUSDT
    assert action.trigger_price == 42_000.0


def test_move_take_profit(exchange):
    account = FuturesAccount(exchange)

    action = account.move_take_profit(BTCUSDT, 55_000.0)

    assert isinstance(action, UpdatePositionTakeProfitAction)
    assert action.exchange is exchange
    assert action.symbol == BTCUSDT
    assert action.trigger_price == 55_000.0


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan")])
def test_move_stop_loss_with_a_price_not_above_zero(exchange, price):
    account = FuturesAccount(exchange)

    with pytest.raises(ValueError, match="`price` must be greater than 0"):
        account.move_stop_loss(BTCUSDT, price)


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan")])
def test_move_take_profit_with_a_price_not_above_zero(exchange, price):
    account = FuturesAccount(exchange)

    with pytest.raises(ValueError, match="`price` must be greater than 0"):
        account.move_take_profit(BTCUSDT, price)


class TestCloseTrackedPosition:
    def test_a_tracked_long_is_closed_with_a_sell_of_its_quantity(self, exchange):
        account = FuturesAccount(exchange)

        exit_order = account.close_tracked_position(
            BTCUSDT, TrackedPosition(PositionSide.LONG, 3.0)
        ).build()

        assert exit_order.symbol == BTCUSDT
        assert exit_order.side == OrderSide.SELL
        assert exit_order.quantity == 3.0
        assert exit_order.reduce_only is False

    def test_a_tracked_short_is_closed_with_a_buy(self, exchange):
        account = FuturesAccount(exchange)

        exit_order = account.close_tracked_position(
            BTCUSDT, TrackedPosition(PositionSide.SHORT, 5.0)
        ).build()

        assert exit_order.side == OrderSide.BUY
        assert exit_order.quantity == 5.0
        assert exit_order.reduce_only is False

    def test_a_closing_ratio_closes_that_share_of_the_tracked_position(self, exchange):
        account = FuturesAccount(exchange)

        exit_order = account.close_tracked_position(
            BTCUSDT, TrackedPosition(PositionSide.LONG, 2.0), closing_ratio=0.25
        ).build()

        assert exit_order.quantity == 0.5


def test_close_position_with_a_ratio_outside_a_share(exchange):
    account = FuturesAccount(exchange)

    with pytest.raises(ValueError, match="`closing_ratio` must be within"):
        account.close_position(BTCUSDT, closing_ratio=1.5)


def test_close_tracked_position_with_a_ratio_outside_a_share(exchange):
    account = FuturesAccount(exchange)

    with pytest.raises(ValueError, match="`closing_ratio` must be within"):
        account.close_tracked_position(
            BTCUSDT, TrackedPosition(PositionSide.LONG, 2.0), 1.5
        )


class TestSnapshot:
    async def test_fetches_only_the_declared_fields(self, exchange):
        exchange.get_open_positions.return_value = {}
        account = FuturesAccount(exchange)

        await account._snapshot(
            AccountRequirement(account=account, positions=True),
            executions_since=WINDOW_START,
        )

        exchange.get_open_positions.assert_awaited_once()
        exchange.get_open_orders.assert_not_awaited()
        exchange.get_balances.assert_not_awaited()
        exchange.get_margin_settings.assert_not_awaited()
        exchange.get_executions_since.assert_not_awaited()
        exchange.get_equity.assert_not_awaited()

    async def test_declared_symbols_scope_every_account_wide_read(self, exchange):
        exchange.get_open_positions.return_value = {}
        exchange.get_open_orders.return_value = []
        exchange.get_balances.return_value = {}
        exchange.get_executions_since.return_value = []
        account = FuturesAccount(exchange)

        await account._snapshot(
            AccountRequirement(
                account=account,
                symbols=(BTCUSDT,),
                positions=True,
                open_orders=True,
                balances=True,
                executions=True,
            ),
            executions_since=WINDOW_START,
        )

        exchange.get_open_positions.assert_awaited_once_with([BTCUSDT])
        exchange.get_open_orders.assert_awaited_once_with([BTCUSDT])
        exchange.get_balances.assert_awaited_once_with([BTCUSDT])
        exchange.get_executions_since.assert_awaited_once_with(WINDOW_START, [BTCUSDT])

    async def test_populates_the_declared_fields(self, exchange):
        balance = Balance.compute(available=800.0, locked=200.0)
        exchange.get_balances.return_value = {"USDT": balance}
        exchange.get_open_orders.return_value = []
        account = FuturesAccount(exchange)

        snapshot = await account._snapshot(
            AccountRequirement(account=account, balances=True, open_orders=True),
            executions_since=WINDOW_START,
        )

        assert snapshot.balance("USDT") == balance
        assert snapshot.open_orders(BTCUSDT) == []

    async def test_snapshot_carries_the_account_name(self, exchange):
        account = FuturesAccount(exchange, name="hedge")

        snapshot = await account._snapshot(
            AccountRequirement(account=account),
            executions_since=WINDOW_START,
        )

        with pytest.raises(StrategyCriticalError, match="hedge"):
            snapshot.position(BTCUSDT)

    async def test_nothing_declared_fetches_nothing(self, exchange):
        account = FuturesAccount(exchange)

        await account._snapshot(
            AccountRequirement(account=account),
            executions_since=WINDOW_START,
        )

        exchange.get_open_positions.assert_not_awaited()
        exchange.get_open_orders.assert_not_awaited()
        exchange.get_balances.assert_not_awaited()

    async def test_fills_window_forwarded_to_the_exchange(self, exchange):
        since = datetime(2026, 5, 1)
        exchange.get_executions_since.return_value = []
        account = FuturesAccount(exchange)

        await account._snapshot(
            AccountRequirement(account=account, executions=True),
            executions_since=since,
        )

        exchange.get_executions_since.assert_awaited_once_with(since, [])

    async def test_report_fills_reads_executions_even_when_undeclared(self, exchange):
        exchange.get_executions_since.return_value = []
        account = FuturesAccount(exchange)

        await account._snapshot(
            AccountRequirement(account=account, symbols=(BTCUSDT,)),
            executions_since=WINDOW_START,
            report_fills=True,
        )

        exchange.get_executions_since.assert_awaited_once_with(WINDOW_START, [BTCUSDT])

    async def test_report_fills_alone_does_not_read_an_account_with_no_symbols(
        self, exchange
    ):
        account = FuturesAccount(exchange)

        await account._snapshot(
            AccountRequirement(account=account),
            executions_since=WINDOW_START,
            report_fills=True,
        )

        exchange.get_executions_since.assert_not_awaited()

    async def test_report_fills_alone_still_raises_on_the_strategy_accessor(
        self, exchange
    ):
        stop_loss = Execution(
            execution_id="sl-1",
            order_id="sl-1",
            symbol=BTCUSDT,
            side=OrderSide.SELL,
            price=90.0,
            quantity=1.0,
            timestamp=WINDOW_START,
            kind="stop-loss",
        )
        exchange.get_executions_since.return_value = [stop_loss]
        account = FuturesAccount(exchange)

        snapshot = await account._snapshot(
            AccountRequirement(account=account, symbols=(BTCUSDT,)),
            executions_since=WINDOW_START,
            report_fills=True,
        )

        with pytest.raises(StrategyCriticalError, match="executions=True"):
            snapshot.stop_loss_fills(BTCUSDT)

    async def test_positions_read_alongside_executions_undeclared(self, exchange):
        exchange.get_executions_since.return_value = []
        account = FuturesAccount(exchange)

        await account._snapshot(
            AccountRequirement(account=account, symbols=(BTCUSDT,)),
            executions_since=WINDOW_START,
            report_fills=True,
        )

        exchange.get_open_positions.assert_awaited_once_with([BTCUSDT])

    async def test_positions_read_for_attribution_stay_undeclared(self, exchange):
        exchange.get_executions_since.return_value = []
        account = FuturesAccount(exchange)

        snapshot = await account._snapshot(
            AccountRequirement(account=account, symbols=(BTCUSDT,)),
            executions_since=WINDOW_START,
            report_fills=True,
        )

        with pytest.raises(StrategyCriticalError, match="positions=True"):
            snapshot.position(BTCUSDT)

    async def test_executions_carry_what_they_did_to_the_position(self, exchange):
        exchange.get_executions_since.return_value = [
            _execution("entry", OrderSide.BUY, 2.0, minute=1),
            _execution("exit", OrderSide.SELL, 1.0, minute=2),
        ]
        exchange.get_open_positions.return_value = {BTCUSDT: _long(1.0)}
        account = FuturesAccount(exchange)

        snapshot = await account._snapshot(
            AccountRequirement(account=account, symbols=(BTCUSDT,), executions=True),
            executions_since=WINDOW_START,
        )

        assert snapshot._effects_by_execution_id() == {
            "entry": "open",
            "exit": "reduce",
        }

    async def test_entry_fills_are_not_part_of_the_snapshot(self, exchange):
        account = FuturesAccount(exchange)

        await account._snapshot(
            AccountRequirement(account=account, notify_entry_fills=True),
            executions_since=WINDOW_START,
        )

        exchange.get_executions_since.assert_not_awaited()

    async def test_executions_queried_on_demand(self, exchange):
        since = datetime(2026, 5, 1)
        exchange.get_executions_since.return_value = []
        account = FuturesAccount(exchange)

        fills = await account._executions_since(since, [BTCUSDT])

        exchange.get_executions_since.assert_awaited_once_with(since, [BTCUSDT])
        assert fills == []

    async def test_every_kind_the_venue_booked_is_returned(self, exchange):
        stop_loss = _execution("stop-1", OrderSide.SELL, 1.0, 1, kind="stop-loss")
        entry = _execution("entry-1", OrderSide.BUY, 1.0, 2, kind="limit")
        exchange.get_executions_since.return_value = [stop_loss, entry]
        account = FuturesAccount(exchange)

        fills = await account._executions_since(datetime(2026, 5, 1), [BTCUSDT])

        assert fills == [stop_loss, entry]

    async def test_equity_fetched_per_declared_currency(self, exchange):
        exchange.get_equity.side_effect = [10_000.0, 9_000.0]
        account = FuturesAccount(exchange)

        snapshot = await account._snapshot(
            AccountRequirement(account=account, equity=("USDT", "USDC")),
            executions_since=WINDOW_START,
        )

        assert snapshot.equity("USDT") == 10_000.0
        assert snapshot.equity("USDC") == 9_000.0

    async def test_recoverable_errors_propagate(self, exchange):
        exchange.get_executions_since.side_effect = ExchangeRecoverableError(
            "rejected read"
        )
        account = FuturesAccount(exchange)

        with pytest.raises(ExchangeRecoverableError, match="rejected read"):
            await account._snapshot(
                AccountRequirement(account=account, executions=True),
                executions_since=datetime(2026, 5, 1),
            )

    async def test_reads_retry_on_transient_errors(self, exchange):
        exchange.get_open_positions.side_effect = [
            ExchangeTransientError("502"),
            {},
        ]
        account = FuturesAccount(exchange)

        snapshot = await account._snapshot(
            AccountRequirement(account=account, positions=True),
            executions_since=WINDOW_START,
            base_delay=0.0,
        )

        assert snapshot.position(BTCUSDT) is None
        assert exchange.get_open_positions.await_count == 2
