from unittest.mock import AsyncMock, Mock

import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import (
    FuturesExchangeProtocol,
    OrderSide,
    PlacedOrder,
    VenueFill,
)
from robottraderslab.futures.futures_market_order import FuturesMarketOrderAction


@pytest.fixture
def mock_exchange() -> FuturesExchangeProtocol:
    async def mock_get_fill() -> VenueFill:
        return VenueFill(
            order_id="test_order_123",
            symbol=Symbol.create("BTC/USDT:USDT"),
            side=OrderSide.BUY,
            quantity=1.0,
        )

    mock = Mock(spec=FuturesExchangeProtocol)
    mock.place_market_order = AsyncMock(
        return_value=PlacedOrder(order_id="test_order_123", get_fill=mock_get_fill)
    )
    return mock


@pytest.fixture
def market_order_action(mock_exchange) -> FuturesMarketOrderAction:
    return FuturesMarketOrderAction(
        exchange=mock_exchange,
        symbol=Symbol.create("BTC/USDT"),
        side=OrderSide.BUY,
        quantity=1.0,
        reduce_only=False,
        take_profit=50000.0,
        stop_loss=40000.0,
        trigger_price=45000.0,
    )


async def test_execute_places_market_order(market_order_action):
    await market_order_action.execute()

    market_order_action.exchange.place_market_order.assert_called_once_with(
        symbol=market_order_action.symbol,
        side=market_order_action.side,
        quantity=market_order_action.quantity,
        reduce_only=market_order_action.reduce_only,
        stop_loss=market_order_action.stop_loss,
        take_profit=market_order_action.take_profit,
        trigger_price=market_order_action.trigger_price,
        client_order_id=market_order_action.client_order_id,
        reason=market_order_action.reason,
        extra_fields=market_order_action.extra_fields,
    )


async def test_exceptions_passthrough_execute(market_order_action):
    error = ValueError("Test error")
    market_order_action.exchange.place_market_order.side_effect = error

    with pytest.raises(type(error)):
        await market_order_action.execute()


async def test_execute_returns_placed_order_and_callback():
    order_fill = VenueFill(
        order_id="order_456",
        symbol=Symbol.create("BTC/USDT"),
        side=OrderSide.BUY,
        quantity=2.5,
    )
    placed_order = PlacedOrder(
        order_id="order_456", get_fill=AsyncMock(return_value=order_fill)
    )
    mock_exchange = Mock(spec=FuturesExchangeProtocol)
    mock_exchange.place_market_order = AsyncMock(return_value=placed_order)

    callback = AsyncMock()
    action = FuturesMarketOrderAction(
        exchange=mock_exchange,
        symbol=Symbol.create("BTC/USDT"),
        side=OrderSide.BUY,
        quantity=1.0,
        on_filled=callback,
    )

    action_result = await action.execute()

    (outcome,) = action_result.orders
    assert outcome.placed_order is placed_order
    assert outcome.on_filled is callback
    assert outcome.placement.kind == "market"
