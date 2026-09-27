import asyncio
from collections.abc import Callable
from unittest.mock import Mock, call

import pytest

from robottraderslab import Symbol
from robottraderslab._core import ActionResult, OrderProtocol
from robottraderslab.exceptions import ExchangeRecoverableError, ExchangeTransientError
from robottraderslab.exchanges import FuturesExchangeProtocol
from robottraderslab.futures.futures_cancel_orders import (
    CancelOrdersForSymbolAction,
)


@pytest.fixture
def mock_exchange() -> Mock:
    return Mock(spec=FuturesExchangeProtocol)


@pytest.fixture
def create_resting_order() -> Callable[[str, str | None], Mock]:
    def _create_resting_order(order_id: str, client_order_id: str | None) -> Mock:
        order = Mock(spec=OrderProtocol)
        order.order_id = order_id
        order.client_order_id = client_order_id
        return order

    return _create_resting_order


@pytest.fixture
def tagged_cancel(
    mock_exchange: Mock, create_resting_order: Callable[[str, str | None], Mock]
) -> CancelOrdersForSymbolAction:
    mock_exchange.get_open_orders.return_value = [
        create_resting_order("rung-1", "run-1-a"),
        create_resting_order("rung-2", "run-2-a"),
    ]
    return CancelOrdersForSymbolAction(
        exchange=mock_exchange, symbol=Symbol.create("BTC/USDT:USDT"), order_tag="a"
    )


@pytest.fixture
def cancel_orders_action(mock_exchange: Mock) -> CancelOrdersForSymbolAction:
    return CancelOrdersForSymbolAction(
        exchange=mock_exchange,
        symbol=Symbol.create("BTC/USDT:USDT"),
    )


async def test_execute_calls_cancel_orders_for_symbol(cancel_orders_action):
    await cancel_orders_action.execute()

    cancel_orders_action.exchange.cancel_orders_for_symbol.assert_called_once_with(
        cancel_orders_action.symbol,
    )


async def test_execute_returns_empty_action_result(cancel_orders_action):
    action_result = await cancel_orders_action.execute()

    assert isinstance(action_result, ActionResult)
    assert action_result.orders == ()


async def test_exceptions_passthrough_execute(cancel_orders_action):
    error = ValueError("Test error")
    cancel_orders_action.exchange.cancel_orders_for_symbol.side_effect = error

    with pytest.raises(type(error)):
        await cancel_orders_action.execute()


async def test_tagged_cancel_withdraws_only_the_orders_carrying_the_tag(
    mock_exchange, create_resting_order
):
    symbol = Symbol.create("BTC/USDT:USDT")
    mock_exchange.get_open_orders.return_value = [
        create_resting_order("rung-1", "run-1-a"),
        create_resting_order("rung-2", "run-2-a"),
        create_resting_order("other-profile", "run-3-b"),
        create_resting_order("untagged", "run-4"),
        create_resting_order("stop-loss", None),
    ]
    action = CancelOrdersForSymbolAction(
        exchange=mock_exchange, symbol=symbol, order_tag="a"
    )

    await action.execute()

    mock_exchange.get_open_orders.assert_awaited_once_with([symbol])
    assert mock_exchange.cancel_order_by_id.await_args_list == [
        call(symbol, "rung-1"),
        call(symbol, "rung-2"),
    ]
    mock_exchange.cancel_orders_for_symbol.assert_not_awaited()


async def test_a_refused_cancel_leaves_the_other_tagged_cancels_standing(
    tagged_cancel, mock_exchange
):
    mock_exchange.cancel_order_by_id.side_effect = [
        ExchangeRecoverableError("order already filled"),
        None,
    ]

    await tagged_cancel.execute()

    assert mock_exchange.cancel_order_by_id.await_count == 2


async def test_a_transient_cancel_raises_once_every_cancel_has_settled(
    tagged_cancel, mock_exchange
):
    settled: list[str] = []

    async def cancel(symbol: Symbol, order_id: str) -> None:
        if order_id == "rung-1":
            raise ExchangeTransientError("timeout")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        settled.append(order_id)

    mock_exchange.cancel_order_by_id.side_effect = cancel

    with pytest.raises(ExchangeTransientError):
        await tagged_cancel.execute()

    assert settled == ["rung-2"]
