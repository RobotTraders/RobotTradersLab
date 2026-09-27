import asyncio
from datetime import datetime

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import CacheFillRecorder
from robottraderslab.exchanges import (
    OrderModifyRequest,
    OrderRequest,
    OrderSide,
    PlacedOrder,
    VenueFill,
)


class TestPlaceOrders:
    async def test_each_order_settles_before_the_next_starts(self, sim, btc_usdt_perp):
        trace: list[str] = []
        queue_one = sim.exchange.place_market_order

        async def traced(symbol, side, quantity, **kwargs):
            trace.append(f"start {quantity}")
            await asyncio.sleep(0)
            trace.append(f"end {quantity}")
            return await queue_one(symbol, side, quantity, **kwargs)

        sim.exchange.place_market_order = traced
        requests = [
            OrderRequest(symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=quantity)
            for quantity in (1.0, 2.0)
        ]

        placed = await sim.exchange.place_orders(requests)

        assert len(placed) == 2
        assert trace == ["start 1.0", "end 1.0", "start 2.0", "end 2.0"]


class TestModifyOrders:
    async def test_each_replacement_settles_before_the_next_starts(
        self, sim, btc_usdt_perp
    ):
        resting = [
            await sim.exchange.place_limit_order(
                symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=quantity, price=100.0
            )
            for quantity in (1.0, 2.0)
        ]
        trace: list[str] = []
        queue_one = sim.exchange.place_limit_order

        async def traced(symbol, side, quantity, price, **kwargs):
            trace.append(f"start {quantity}")
            await asyncio.sleep(0)
            trace.append(f"end {quantity}")
            return await queue_one(symbol, side, quantity, price, **kwargs)

        sim.exchange.place_limit_order = traced
        requests = [
            OrderModifyRequest(
                order_id=order.order_id,
                order=OrderRequest(
                    symbol=btc_usdt_perp,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    limit_price=200.0,
                ),
            )
            for order, quantity in zip(resting, (1.0, 2.0))
        ]

        replaced = await sim.exchange.modify_orders(requests)

        assert len(replaced) == 2
        assert trace == ["start 1.0", "end 1.0", "start 2.0", "end 2.0"]


class TestGetExecutionsSince:
    async def test_simulator_reports_no_executions(self, sim):
        executions = await sim.exchange.get_executions_since(datetime.now(), [])

        assert executions == []


class TestSimlatedExchange:
    async def test_place_market_order_returns_placed_order_with_id(
        self, sim, btc_usdt_perp: Symbol
    ):
        placed_order = await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
        )

        assert isinstance(placed_order, PlacedOrder)
        assert isinstance(placed_order.order_id, str)
        assert len(placed_order.order_id) > 0

    async def test_place_limit_order_returns_placed_order_with_id(
        self, sim, btc_usdt_perp: Symbol
    ):
        placed_order = await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            price=5000.0,
        )

        assert isinstance(placed_order, PlacedOrder)
        assert isinstance(placed_order.order_id, str)
        assert len(placed_order.order_id) > 0

    async def test_multiple_orders_have_unique_ids(self, sim, btc_usdt_perp: Symbol):
        order1 = await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
        )

        order2 = await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
        )

        order3 = await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=1.5,
            price=600.0,
        )

        ids = [order1.order_id, order2.order_id, order3.order_id]
        assert len(set(ids)) == len(ids)

    async def test_place_market_order_provides_get_fill_callback(
        self, sim, btc_usdt_perp: Symbol
    ):
        placed_order = await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.5,
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        order_fill = await placed_order.get_fill()

        assert isinstance(order_fill, VenueFill)
        assert order_fill.order_id == placed_order.order_id

    async def test_a_market_order_still_resting_reports_no_fill(
        self, sim, btc_usdt_perp: Symbol
    ):
        placed_order = await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.5,
        )

        order_fill = await placed_order.get_fill()

        assert order_fill is None

    async def test_place_limit_order_provides_get_fill_callback(
        self, sim, btc_usdt_perp: Symbol
    ):
        placed_order = await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=2.5,
            price=500.0,
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=80.0, high=120.0, close=100.0)

        order_fill = await placed_order.get_fill()

        assert order_fill is None


_TAG_COLUMN = CacheFillRecorder.HEADER.index("tag")


class TestFillTags:
    async def test_a_market_orders_fill_carries_the_tag_its_id_encodes(
        self, sim, btc_usdt_perp: Symbol
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            client_order_id="fakerun99-7-1h-alpha",
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        recorded_tags = {fill[_TAG_COLUMN] for fill in sim.fill_recorder.fills}
        assert recorded_tags == {"1h-alpha"}

    async def test_a_trigger_orders_fill_carries_the_tag_its_id_encodes(
        self, sim, btc_usdt_perp: Symbol
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            trigger_price=90.0,
            client_order_id="fakerun99-8-r2",
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0, low=80.0)

        recorded_tags = {fill[_TAG_COLUMN] for fill in sim.fill_recorder.fills}
        assert recorded_tags == {"r2"}

    async def test_an_order_placed_without_an_id_records_no_tag(
        self, sim, btc_usdt_perp: Symbol
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        recorded_tags = {fill[_TAG_COLUMN] for fill in sim.fill_recorder.fills}
        assert recorded_tags == {None}
