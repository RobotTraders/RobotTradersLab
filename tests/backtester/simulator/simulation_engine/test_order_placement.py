from robottraderslab import Symbol
from robottraderslab.exchanges import OrderSide


class TestOrderPlacement:
    async def test_place_limit_buy_order(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=2.0, price=100.0
        )

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0
        assert sim.open_orders[0].kind == "limit"
        assert sim.open_orders[0].symbol == btc_usdt_perp
        assert sim.open_orders[0].side == OrderSide.BUY
        assert sim.open_orders[0].stop_loss is None
        assert sim.open_orders[0].take_profit is None

    async def test_resting_order_reports_its_client_order_id(
        self, sim, btc_usdt_perp: Symbol
    ):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            price=100.0,
            client_order_id="fakeprocess999-r2",
        )

        assert sim.open_orders[0].client_order_id == "fakeprocess999-r2"

    async def test_resting_trigger_order_reports_its_client_order_id(
        self, sim, btc_usdt_perp: Symbol
    ):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            price=100.0,
            trigger_price=90.0,
            client_order_id="fakeprocess999-r3",
        )

        assert sim.open_orders[0].client_order_id == "fakeprocess999-r3"

    async def test_place_limit_sell_order(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=2.0, price=100.0
        )

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0
        assert sim.open_orders[0].kind == "limit"
        assert sim.open_orders[0].symbol == btc_usdt_perp
        assert sim.open_orders[0].side == OrderSide.SELL
        assert sim.open_orders[0].stop_loss is None
        assert sim.open_orders[0].take_profit is None

    async def test_place_limit_order_with_stop_loss(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            price=100.0,
            stop_loss=50.0,
        )

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0
        assert sim.open_orders[0].kind == "limit"
        assert sim.open_orders[0].symbol == btc_usdt_perp
        assert sim.open_orders[0].side == OrderSide.BUY
        assert sim.open_orders[0].stop_loss == 50.0
        assert sim.open_orders[0].take_profit is None

    async def test_place_limit_order_with_take_profit(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            price=100.0,
            take_profit=150.0,
        )

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0
        assert sim.open_orders[0].kind == "limit"
        assert sim.open_orders[0].symbol == btc_usdt_perp
        assert sim.open_orders[0].side == OrderSide.BUY
        assert sim.open_orders[0].stop_loss is None
        assert sim.open_orders[0].take_profit == 150.0

    async def test_place_triggered_limit_order(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            price=100.0,
            trigger_price=150.0,
        )

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0
        assert sim.open_orders[0].kind == "trigger"
        assert sim.open_orders[0].symbol == btc_usdt_perp
        assert sim.open_orders[0].order.kind == "limit"

    async def test_place_many_limit_orders(self, sim, btc_usdt_perp: Symbol):
        nb = 3
        for _ in range(nb):
            await sim.exchange.place_limit_order(
                symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=2.0, price=100.0
            )

        assert len(sim.open_orders) == nb
        assert len(sim.open_positions) == 0
        for order in sim.open_orders:
            assert order.kind == "limit"
            assert order.symbol == btc_usdt_perp
            assert order.side == OrderSide.BUY
            assert order.stop_loss is None
            assert order.take_profit is None

    async def test_place_market_buy_order(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=2.0
        )

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0
        assert sim.open_orders[0].kind == "market"
        assert sim.open_orders[0].symbol == btc_usdt_perp
        assert sim.open_orders[0].side == OrderSide.BUY
        assert sim.open_orders[0].stop_loss is None
        assert sim.open_orders[0].take_profit is None

    async def test_place_market_sell_order(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=2.0
        )

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0
        assert sim.open_orders[0].kind == "market"
        assert sim.open_orders[0].symbol == btc_usdt_perp
        assert sim.open_orders[0].side == OrderSide.SELL
        assert sim.open_orders[0].stop_loss is None
        assert sim.open_orders[0].take_profit is None

    async def test_place_market_order_with_stop_loss(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            stop_loss=50.0,
        )

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0
        assert sim.open_orders[0].kind == "market"
        assert sim.open_orders[0].symbol == btc_usdt_perp
        assert sim.open_orders[0].side == OrderSide.BUY
        assert sim.open_orders[0].stop_loss == 50.0
        assert sim.open_orders[0].take_profit is None

    async def test_place_market_order_with_take_profit(
        self, sim, btc_usdt_perp: Symbol
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            take_profit=150.0,
        )

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0
        assert sim.open_orders[0].kind == "market"
        assert sim.open_orders[0].symbol == btc_usdt_perp
        assert sim.open_orders[0].side == OrderSide.BUY
        assert sim.open_orders[0].stop_loss is None
        assert sim.open_orders[0].take_profit == 150.0

    async def test_place_triggered_market_order(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            trigger_price=150.0,
        )

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0
        assert sim.open_orders[0].kind == "trigger"
        assert sim.open_orders[0].symbol == btc_usdt_perp
        assert sim.open_orders[0].order.kind == "market"

    async def test_place_many_market_orders(self, sim, btc_usdt_perp: Symbol):
        nb = 3
        for _ in range(nb):
            await sim.exchange.place_market_order(
                symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=2.0
            )

        assert len(sim.open_orders) == nb
        assert len(sim.open_positions) == 0
        for order in sim.open_orders:
            assert order.kind == "market"
            assert order.symbol == btc_usdt_perp
            assert order.side == OrderSide.BUY
            assert order.stop_loss is None
            assert order.take_profit is None
