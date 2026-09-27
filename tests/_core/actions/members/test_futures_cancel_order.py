from unittest.mock import AsyncMock, Mock

import pytest

from robottraderslab import Symbol
from robottraderslab._core import ActionResult
from robottraderslab.exchanges import FuturesExchangeProtocol
from robottraderslab.futures.futures_cancel_order import CancelOrderByIdAction


@pytest.fixture
def mock_exchange() -> FuturesExchangeProtocol:
    mock = Mock(spec=FuturesExchangeProtocol)
    mock.cancel_order_by_id = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def cancel_order_action(mock_exchange) -> CancelOrderByIdAction:
    return CancelOrderByIdAction(
        exchange=mock_exchange,
        symbol=Symbol("BTC", "USDT", "USDT"),
        order_id="order-123",
    )


async def test_execute_calls_cancel_order_by_id(cancel_order_action):
    await cancel_order_action.execute()

    cancel_order_action.exchange.cancel_order_by_id.assert_called_once_with(
        cancel_order_action.symbol,
        cancel_order_action.order_id,
    )


async def test_execute_returns_empty_action_result(cancel_order_action):
    action_result = await cancel_order_action.execute()

    assert isinstance(action_result, ActionResult)
    assert action_result.orders == ()


async def test_exceptions_passthrough_execute(cancel_order_action):
    error = ValueError("Test error")
    cancel_order_action.exchange.cancel_order_by_id.side_effect = error

    with pytest.raises(type(error)):
        await cancel_order_action.execute()
