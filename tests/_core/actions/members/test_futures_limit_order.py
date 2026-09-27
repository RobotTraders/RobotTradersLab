from unittest.mock import AsyncMock, Mock

import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import FuturesExchangeProtocol, OrderSide, PlacedOrder
from robottraderslab.futures.futures_limit_order import FuturesLimitOrderAction


@pytest.fixture
def mock_exchange() -> FuturesExchangeProtocol:
    mock = Mock(spec=FuturesExchangeProtocol)
    mock.place_limit_order = AsyncMock(
        return_value=PlacedOrder(order_id="test_order_456")
    )
    return mock


@pytest.fixture
def limit_order_action(mock_exchange) -> FuturesLimitOrderAction:
    return FuturesLimitOrderAction(
        exchange=mock_exchange,
        symbol=Symbol.create("BTC/USDT"),
        side=OrderSide.BUY,
        quantity=1.0,
        price=50000.0,
        reduce_only=False,
        take_profit=60000.0,
        stop_loss=45000.0,
        trigger_price=51000.0,
    )


async def test_execute_places_limit_order(limit_order_action):
    await limit_order_action.execute()

    limit_order_action.exchange.place_limit_order.assert_called_once_with(
        symbol=limit_order_action.symbol,
        side=limit_order_action.side,
        quantity=limit_order_action.quantity,
        price=limit_order_action.price,
        reduce_only=limit_order_action.reduce_only,
        stop_loss=limit_order_action.stop_loss,
        take_profit=limit_order_action.take_profit,
        trigger_price=limit_order_action.trigger_price,
        client_order_id=limit_order_action.client_order_id,
        time_in_force=limit_order_action.time_in_force,
        reason=limit_order_action.reason,
        extra_fields=limit_order_action.extra_fields,
    )


async def test_exceptions_passthrough_execute(limit_order_action):
    error = ValueError("Test error")
    limit_order_action.exchange.place_limit_order.side_effect = error

    with pytest.raises(type(error)):
        await limit_order_action.execute()


async def test_execute_returns_placed_order_and_callback():
    placed_order = PlacedOrder(order_id="order_456")
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_limit_order = AsyncMock(return_value=placed_order)

    callback = AsyncMock()
    action = FuturesLimitOrderAction(
        exchange=mock_exchange,
        symbol=Symbol.create("BTC/USDT"),
        side=OrderSide.BUY,
        quantity=1.0,
        price=4.0,
        on_filled=callback,
    )

    action_result = await action.execute()

    (outcome,) = action_result.orders
    assert outcome.placed_order is placed_order
    assert outcome.on_filled is callback
    assert outcome.placement.kind == "limit"


async def test_placement_carries_the_reason_the_strategy_gave():
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_limit_order = AsyncMock(
        return_value=PlacedOrder(order_id="order_789")
    )
    action = FuturesLimitOrderAction(
        exchange=mock_exchange,
        symbol=Symbol.create("BTC/USDT"),
        side=OrderSide.BUY,
        quantity=1.0,
        price=4.0,
        reason="resting at the band",
    )

    action_result = await action.execute()

    (outcome,) = action_result.orders
    assert outcome.placement.reason == "resting at the band"
