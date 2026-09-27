import asyncio
import logging
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, Mock, patch

import pytest

from robottraderslab import Symbol
from robottraderslab._core import (
    ActionResult,
    BaseExchangeAction,
    BookKeeper,
    execute_trading_actions,
)
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ExchangeTransientError,
)
from robottraderslab.exchanges import (
    FuturesExchangeProtocol,
    OrderFill,
    OrderPlacement,
    OrderSide,
    OrderType,
    PlacedOrder,
    VenueFill,
)
from robottraderslab.futures.futures_limit_order import FuturesLimitOrderAction
from robottraderslab.futures.futures_market_order import FuturesMarketOrderAction


def stamped(fill: VenueFill, kind: OrderType) -> OrderFill:
    return OrderFill(
        order_id=fill.order_id,
        symbol=fill.symbol,
        side=fill.side,
        quantity=fill.quantity,
        kind=kind,
        timestamp=fill.timestamp,
        filled_value=fill.filled_value,
        source="strategy",
    )


async def test_executes_all_actions():
    get_fill = AsyncMock(
        return_value=VenueFill(
            order_id="test",
            symbol=Symbol.create("BTC/USDT"),
            side=OrderSide.BUY,
            quantity=1.0,
        )
    )
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_market_order = AsyncMock(
        return_value=PlacedOrder(order_id="test", get_fill=get_fill)
    )
    mock_exchange.place_limit_order = AsyncMock(
        return_value=PlacedOrder(order_id="test")
    )

    sample_actions: list[BaseExchangeAction] = [
        FuturesMarketOrderAction(
            exchange=mock_exchange,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
        ),
        FuturesLimitOrderAction(
            exchange=mock_exchange,
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            quantity=2.0,
            price=2000.0,
        ),
    ]

    await execute_trading_actions(sample_actions)

    mock_exchange.place_market_order.assert_called_once()
    mock_exchange.place_limit_order.assert_called_once()


async def test_executes_all_actions_even_if_one_action_fails():
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_market_order = AsyncMock(side_effect=ValueError("Test error"))
    mock_exchange.place_limit_order = AsyncMock(
        return_value=PlacedOrder(order_id="test")
    )

    sample_actions: list[BaseExchangeAction] = [
        FuturesMarketOrderAction(
            exchange=mock_exchange,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
        ),
        FuturesLimitOrderAction(
            exchange=mock_exchange,
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            quantity=2.0,
            price=2000.0,
        ),
    ]

    await execute_trading_actions(sample_actions)

    mock_exchange.place_market_order.assert_called_once()
    mock_exchange.place_limit_order.assert_called_once()


async def test_with_empty_list():
    await execute_trading_actions([])


async def test_internal_error_log_names_action_and_symbol(caplog):
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_market_order = AsyncMock(side_effect=ValueError("boom"))
    action = FuturesMarketOrderAction(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
    )

    with caplog.at_level(logging.ERROR):
        await execute_trading_actions([action])

    assert "Internal error in FuturesMarketOrderAction (BTC/USDT): boom" in caplog.text


async def test_rejection_log_names_action_and_symbol(caplog):
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_market_order = AsyncMock(
        side_effect=ExchangeRecoverableError("Insufficient balance")
    )
    action = FuturesMarketOrderAction(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
    )

    with caplog.at_level(logging.WARNING):
        await execute_trading_actions([action])

    rejection = next(
        record for record in caplog.records if "Exchange rejected" in record.message
    )
    assert rejection.levelno == logging.WARNING
    assert rejection.getMessage() == (
        "Exchange rejected FuturesMarketOrderAction (BTC/USDT): Insufficient balance"
    )


async def test_continues_execution_on_exchange_warning_error():
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_market_order = AsyncMock(
        side_effect=ExchangeRecoverableError("Insufficient balance")
    )
    mock_exchange.place_limit_order = AsyncMock(
        return_value=PlacedOrder(order_id="test")
    )

    sample_actions: list[BaseExchangeAction] = [
        FuturesMarketOrderAction(
            exchange=mock_exchange,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
        ),
        FuturesLimitOrderAction(
            exchange=mock_exchange,
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            quantity=2.0,
            price=2000.0,
        ),
    ]

    await execute_trading_actions(sample_actions)

    mock_exchange.place_market_order.assert_called_once()
    mock_exchange.place_limit_order.assert_called_once()


async def test_with_critical_error_during_execution():
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_market_order = AsyncMock(
        side_effect=ExchangeCriticalError("Critical failure")
    )

    sample_actions: list[BaseExchangeAction] = [
        FuturesMarketOrderAction(
            exchange=mock_exchange,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
        ),
    ]

    with pytest.raises(ExchangeCriticalError, match="Critical failure"):
        await execute_trading_actions(sample_actions)


async def test_execution_callbacks_with_executed_market_order():
    plugin_fill = VenueFill(
        order_id="test",
        symbol=Symbol.create("BTC/USDT"),
        side=OrderSide.BUY,
        quantity=1.0,
    )
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_market_order = AsyncMock(
        return_value=PlacedOrder(
            order_id="test",
            get_fill=AsyncMock(return_value=plugin_fill),
        )
    )
    action = FuturesMarketOrderAction(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
    )
    execution_callback = AsyncMock()

    post_execution_callbacks = await execute_trading_actions(
        [action],
        on_order_filled=[execution_callback],
    )
    for post_callback in post_execution_callbacks:
        await post_callback()

    execution_callback.assert_awaited_once_with(stamped(plugin_fill, "market"))


async def test_execution_callbacks_with_executed_limit_order():
    plugin_fill = VenueFill(
        order_id="test",
        symbol=Symbol.create("BTC/USDT"),
        side=OrderSide.BUY,
        quantity=1.0,
    )
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_limit_order = AsyncMock(
        return_value=PlacedOrder(
            order_id="test",
            get_fill=AsyncMock(return_value=plugin_fill),
        )
    )
    action = FuturesLimitOrderAction(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
        price=50000.0,
    )
    execution_callback = AsyncMock()

    post_execution_callbacks = await execute_trading_actions(
        [action],
        on_order_filled=[execution_callback],
    )
    for post_callback in post_execution_callbacks:
        await post_callback()

    execution_callback.assert_awaited_once_with(stamped(plugin_fill, "limit"))


async def test_execution_callbacks_with_none_executed_order():
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_limit_order = AsyncMock(
        return_value=PlacedOrder(
            order_id="test",
            get_fill=AsyncMock(return_value=None),
        )
    )
    action = FuturesLimitOrderAction(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
        price=50000.0,
    )
    execution_callback = AsyncMock()

    post_execution_callbacks = await execute_trading_actions(
        [action],
        on_order_filled=[execution_callback],
    )
    for post_callback in post_execution_callbacks:
        await post_callback()

    execution_callback.assert_not_awaited()


async def test_execution_callback_with_failing_callback():
    plugin_fill = VenueFill(
        order_id="test",
        symbol=Symbol.create("BTC/USDT"),
        side=OrderSide.BUY,
        quantity=1.0,
    )
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_market_order = AsyncMock(
        return_value=PlacedOrder(
            order_id="test",
            get_fill=AsyncMock(return_value=plugin_fill),
        )
    )
    action = FuturesMarketOrderAction(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
    )
    failing_callback = AsyncMock(side_effect=RuntimeError("boom"))
    passing_callback = AsyncMock()

    post_execution_callbacks = await execute_trading_actions(
        [action],
        on_order_filled=[failing_callback, passing_callback],
    )
    for post_callback in post_execution_callbacks:
        await post_callback()

    stamped_fill = stamped(plugin_fill, "market")
    failing_callback.assert_awaited_once_with(stamped_fill)
    passing_callback.assert_awaited_once_with(stamped_fill)


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_transient_error_with_recovery(mock_sleep):
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_market_order = AsyncMock(
        side_effect=[
            ExchangeTransientError("502"),
            PlacedOrder(order_id="test"),
        ]
    )
    action = FuturesMarketOrderAction(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
    )

    await execute_trading_actions([action])

    assert mock_exchange.place_market_order.call_count == 2


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_transient_error_exhausted(mock_sleep, caplog):
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_market_order = AsyncMock(
        side_effect=ExchangeTransientError("502")
    )
    mock_exchange.place_limit_order = AsyncMock(
        return_value=PlacedOrder(order_id="test")
    )

    actions: list[BaseExchangeAction] = [
        FuturesMarketOrderAction(
            exchange=mock_exchange,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
        ),
        FuturesLimitOrderAction(
            exchange=mock_exchange,
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            quantity=2.0,
            price=2000.0,
        ),
    ]

    with caplog.at_level(logging.ERROR):
        await execute_trading_actions(actions, max_attempts=2)

    assert mock_exchange.place_market_order.call_count == 2
    mock_exchange.place_limit_order.assert_called_once()
    skip = next(
        record for record in caplog.records if "skipped after" in record.message
    )
    assert skip.levelno == logging.ERROR
    assert skip.getMessage() == (
        "FuturesMarketOrderAction (BTC/USDT) skipped after 2 attempts: 502"
    )


@patch("robottraderslab._core.retry.random.uniform", side_effect=lambda low, high: high)
@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_respects_custom_retry_params(mock_sleep, mock_uniform):
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_market_order = AsyncMock(
        side_effect=ExchangeTransientError("502")
    )

    action = FuturesMarketOrderAction(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
    )

    await execute_trading_actions(
        [action],
        max_attempts=2,
        base_delay=5.0,
    )

    assert mock_exchange.place_market_order.call_count == 2
    mock_sleep.assert_called_once_with(5.0)


@dataclass(kw_only=True, slots=True)
class RecordedAction(BaseExchangeAction):
    label: str
    log: list[str]
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event | None = None
    error: Exception | None = None

    async def execute(self) -> ActionResult:
        self.log.append(f"start {self.label}")
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        self.log.append(f"end {self.label}")
        return ActionResult()


class TestDeclaredDependencies:
    BTC = Symbol.create("BTC/USDT")
    ETH = Symbol.create("ETH/USDT")

    def _book(self, *actions):
        bookkeeper = BookKeeper()
        for action, after in actions:
            bookkeeper.add(action, after=after)
        return bookkeeper

    async def test_independent_actions_run_at_the_same_time(self):
        log: list[str] = []
        release = asyncio.Event()
        blocked = RecordedAction(symbol=self.BTC, label="btc", log=log, release=release)
        other = RecordedAction(symbol=self.ETH, label="eth", log=log)
        bookkeeper = self._book((blocked, ()), (other, ()))

        execution = asyncio.create_task(
            execute_trading_actions(
                bookkeeper.list_actions(), declared_waits=bookkeeper.declared_waits()
            )
        )
        await other.started.wait()
        release.set()
        await execution

        assert log == ["start btc", "start eth", "end eth", "end btc"]

    async def test_dependent_action_waits_for_what_it_declared(self):
        log: list[str] = []
        first = RecordedAction(symbol=self.BTC, label="cancel", log=log)
        second = RecordedAction(symbol=self.BTC, label="place", log=log)
        bookkeeper = self._book((first, ()), (second, first))

        await execute_trading_actions(
            bookkeeper.list_actions(), declared_waits=bookkeeper.declared_waits()
        )

        assert log == ["start cancel", "end cancel", "start place", "end place"]

    async def test_action_waits_for_every_declared_dependency(self):
        log: list[str] = []
        first = RecordedAction(symbol=self.BTC, label="btc", log=log)
        second = RecordedAction(symbol=self.ETH, label="eth", log=log)
        last = RecordedAction(symbol=self.BTC, label="place", log=log)
        bookkeeper = self._book((first, ()), (second, ()), (last, [first, second]))

        await execute_trading_actions(
            bookkeeper.list_actions(), declared_waits=bookkeeper.declared_waits()
        )

        assert log.index("start place") > log.index("end eth")
        assert log.index("start place") > log.index("end btc")

    async def test_undeclared_action_waits_for_everything_before_it(self):
        log: list[str] = []
        independent = RecordedAction(symbol=self.BTC, label="btc", log=log)
        undeclared = RecordedAction(symbol=self.ETH, label="eth", log=log)
        bookkeeper = self._book((independent, ()), (undeclared, None))

        await execute_trading_actions(
            bookkeeper.list_actions(), declared_waits=bookkeeper.declared_waits()
        )

        assert log == ["start btc", "end btc", "start eth", "end eth"]

    async def test_rejected_dependency_does_not_block_its_followers(self):
        log: list[str] = []
        rejected = RecordedAction(
            symbol=self.BTC,
            label="cancel",
            log=log,
            error=ExchangeRecoverableError("no order"),
        )
        follower = RecordedAction(symbol=self.BTC, label="place", log=log)
        bookkeeper = self._book((rejected, ()), (follower, rejected))

        await execute_trading_actions(
            bookkeeper.list_actions(), declared_waits=bookkeeper.declared_waits()
        )

        assert "end place" in log

    async def test_critical_error_waits_for_the_actions_in_flight(self):
        log: list[str] = []
        release = asyncio.Event()
        critical = RecordedAction(
            symbol=self.BTC,
            label="btc",
            log=log,
            error=ExchangeCriticalError("auth"),
        )
        slow = RecordedAction(symbol=self.ETH, label="eth", log=log, release=release)
        bookkeeper = self._book((critical, ()), (slow, ()))

        execution = asyncio.create_task(
            execute_trading_actions(
                bookkeeper.list_actions(), declared_waits=bookkeeper.declared_waits()
            )
        )
        await slow.started.wait()
        release.set()

        with pytest.raises(ExchangeCriticalError):
            await execution

        assert "end eth" in log


async def test_an_accepted_limit_order_is_reported_as_placed():
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_limit_order = AsyncMock(
        return_value=PlacedOrder(order_id="venue-1")
    )
    action = FuturesLimitOrderAction(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
        price=90_000.0,
    )
    placement_callback = AsyncMock()

    for post_callback in await execute_trading_actions(
        [action], on_order_placed=[placement_callback]
    ):
        await post_callback()

    placement_callback.assert_awaited_once_with(
        OrderPlacement(
            order_id="venue-1",
            symbol=Symbol.create("BTC/USDT"),
            side=OrderSide.BUY,
            quantity=1.0,
            kind="limit",
            price=90_000.0,
            client_order_id=action.client_order_id,
        )
    )


async def test_an_order_the_venue_rejected_is_not_reported_as_placed():
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_limit_order = AsyncMock(
        side_effect=ExchangeRecoverableError("rejected")
    )
    action = FuturesLimitOrderAction(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
        price=90_000.0,
    )
    placement_callback = AsyncMock()

    for post_callback in await execute_trading_actions(
        [action], on_order_placed=[placement_callback]
    ):
        await post_callback()

    placement_callback.assert_not_awaited()


async def test_a_fill_carries_the_id_the_venue_reports():
    plugin_fill = VenueFill(
        order_id="venue-1",
        symbol=Symbol.create("BTC/USDT"),
        side=OrderSide.BUY,
        quantity=1.0,
        client_order_id="chosen-by-the-caller",
    )
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_market_order = AsyncMock(
        return_value=PlacedOrder(
            order_id="venue-1",
            get_fill=AsyncMock(return_value=plugin_fill),
        )
    )
    action = FuturesMarketOrderAction(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
    )
    fill_callback = AsyncMock()

    for post_callback in await execute_trading_actions(
        [action], on_order_filled=[fill_callback]
    ):
        await post_callback()

    reported = fill_callback.await_args.args[0]
    assert reported.client_order_id == "chosen-by-the-caller"


async def test_a_placement_and_its_fill_carry_one_client_order_id():
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    action = FuturesLimitOrderAction(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
        price=90_000.0,
    )
    plugin_fill = VenueFill(
        order_id="venue-1",
        symbol=Symbol.create("BTC/USDT"),
        side=OrderSide.BUY,
        quantity=1.0,
        client_order_id=action.client_order_id,
    )
    mock_exchange.place_limit_order = AsyncMock(
        return_value=PlacedOrder(
            order_id="venue-1",
            get_fill=AsyncMock(return_value=plugin_fill),
        )
    )
    placement_callback = AsyncMock()
    fill_callback = AsyncMock()

    for post_callback in await execute_trading_actions(
        [action],
        on_order_filled=[fill_callback],
        on_order_placed=[placement_callback],
    ):
        await post_callback()

    placement = placement_callback.await_args.args[0]
    fill = fill_callback.await_args.args[0]
    assert placement.client_order_id == fill.client_order_id == action.client_order_id


async def test_a_placement_and_its_fill_carry_the_reason_the_strategy_gave():
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    action = FuturesMarketOrderAction(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        quantity=1.0,
        reason="impulse long exit",
    )
    plugin_fill = VenueFill(
        order_id="venue-1",
        symbol=Symbol.create("BTC/USDT"),
        side=OrderSide.SELL,
        quantity=1.0,
    )
    mock_exchange.place_market_order = AsyncMock(
        return_value=PlacedOrder(
            order_id="venue-1",
            get_fill=AsyncMock(return_value=plugin_fill),
        )
    )
    placement_callback = AsyncMock()
    fill_callback = AsyncMock()

    for post_callback in await execute_trading_actions(
        [action],
        on_order_filled=[fill_callback],
        on_order_placed=[placement_callback],
    ):
        await post_callback()

    placement = placement_callback.await_args.args[0]
    fill = fill_callback.await_args.args[0]
    assert placement.reason == fill.reason == "impulse long exit"


async def test_a_venue_reporting_no_client_order_id_reports_none_on_both_events():
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    action = FuturesMarketOrderAction(
        exchange=mock_exchange,
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=1.0,
    )
    plugin_fill = VenueFill(
        order_id="venue-1",
        symbol=Symbol.create("BTC/USDT"),
        side=OrderSide.BUY,
        quantity=1.0,
    )
    mock_exchange.place_market_order = AsyncMock(
        return_value=PlacedOrder(
            order_id="venue-1",
            get_fill=AsyncMock(return_value=plugin_fill),
        )
    )
    placement_callback = AsyncMock()
    fill_callback = AsyncMock()

    for post_callback in await execute_trading_actions(
        [action],
        on_order_filled=[fill_callback],
        on_order_placed=[placement_callback],
    ):
        await post_callback()

    assert fill_callback.await_args.args[0].client_order_id is None
