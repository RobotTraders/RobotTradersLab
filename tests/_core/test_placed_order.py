from robottraderslab import Symbol
from robottraderslab.exchanges import OrderSide, PlacedOrder, VenueFill


async def test_placed_order_fetches_details_via_callback():
    async def mock_get_fill():
        return VenueFill(
            order_id="test_123",
            symbol=Symbol.create("BTC/USDT:USDT"),
            side=OrderSide.BUY,
            quantity=1.5,
        )

    placed_order = PlacedOrder(order_id="test_123", get_fill=mock_get_fill)

    order_fill = await placed_order.get_fill()

    assert order_fill.order_id == "test_123"
    assert order_fill.quantity == 1.5


async def test_placed_order_has_default_callback():
    placed_order = PlacedOrder(order_id="test_123")

    order_fill = await placed_order.get_fill()

    assert order_fill is None
