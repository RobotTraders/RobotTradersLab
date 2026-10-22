import asyncio
from datetime import datetime

import pytest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    SimulatedFuturesExchange,
)
from robottraderslab.exceptions import StrategyCriticalError, SymbolNotListedError
from robottraderslab.exchanges import (
    Instrument,
    OrderModifyRequest,
    OrderRequest,
    OrderSide,
    PlacedOrder,
    Quote,
    TriggerType,
    VenueFill,
)
from robottraderslab.futures.futures_market_order import FuturesMarketOrderAction


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


class TestGetQuote:
    async def test_the_close_of_the_candle_stood_on_is_bid_and_ask(
        self, sim, btc_usdt_perp
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=101.5)

        quote = await sim.exchange.get_quote(btc_usdt_perp)

        assert quote == Quote(bid=101.5, ask=101.5)

    async def test_the_next_candle_moves_the_quote(self, sim, btc_usdt_perp):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=101.5)
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=99.0)

        quote = await sim.exchange.get_quote(btc_usdt_perp)

        assert quote == Quote(bid=99.0, ask=99.0)

    async def test_a_symbol_with_no_candle_seen(self, sim, btc_usdt_perp):
        with pytest.raises(SymbolNotListedError):
            await sim.exchange.get_quote(btc_usdt_perp)


class TestInstrument:
    async def test_the_step_an_order_is_cut_to(self, sim, btc_usdt_perp):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=101.5)

        instrument = await sim.exchange.get_instrument(btc_usdt_perp)

        assert instrument == Instrument(
            quantity_step=1e-8,
            minimum_order_value=None,
            maximum_leverage=None,
            maximum_order_quantity=None,
            maximum_market_order_quantity=None,
        )

    async def test_the_rules_of_a_symbol_with_no_candle_seen(self, sim, btc_usdt_perp):
        instrument = await sim.exchange.get_instrument(btc_usdt_perp)

        assert instrument.quantity_step == 1e-8

    async def test_a_quantity_cut_down_to_its_step(self, sim, btc_usdt_perp):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=101.5)

        rounded_quantity = await sim.exchange.round_quantity(
            btc_usdt_perp, 0.123456789923
        )

        assert rounded_quantity == 0.12345678

    async def test_a_price_left_as_it_is(self, sim, btc_usdt_perp):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=101.5)

        rounded_price = await sim.exchange.round_price(btc_usdt_perp, 101.123456789)

        assert rounded_price == 101.123456789

    @pytest.mark.parametrize(
        ("read", "arguments"),
        [("round_price", (1.0,)), ("round_quantity", (1.0,))],
    )
    async def test_a_symbol_with_no_candle_seen(
        self, sim, btc_usdt_perp, read, arguments
    ):
        with pytest.raises(SymbolNotListedError):
            await getattr(sim.exchange, read)(btc_usdt_perp, *arguments)


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

    async def test_a_market_order_carrying_venue_params_fills(self, sim, btc_usdt_perp):
        order = FuturesMarketOrderAction(
            exchange=sim.exchange,
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.5,
            params={"bitget": {"venueOnly": "99"}},
        )
        (outcome,) = (await order.execute()).orders
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        order_fill = await outcome.placed_order.get_fill()

        assert isinstance(order_fill, VenueFill)
        assert order_fill.quantity == 1.5

    async def test_a_trigger_watching_the_index_fills_on_the_candle(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            trigger_price=90.0,
            trigger_type=TriggerType.INDEX_PRICE,
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0, low=80.0)

        assert len(sim.fill_recorder.fills) == 1

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


def test_declares_in_place_modification_of_resting_orders():
    assert SimulatedFuturesExchange.capabilities.modifies_resting_orders_in_place


def test_declares_no_trailing_stop_order():
    assert not SimulatedFuturesExchange.capabilities.places_trailing_stop_orders


async def test_trailing_stop_order_requested_on_the_simulator(sim, btc_usdt_perp):
    with pytest.raises(StrategyCriticalError, match="simulator"):
        await sim.exchange.place_trailing_stop_order(btc_usdt_perp, distance=5.0)
