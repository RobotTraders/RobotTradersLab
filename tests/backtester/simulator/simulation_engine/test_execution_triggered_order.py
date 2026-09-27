from robottraderslab import Symbol
from robottraderslab.exchanges import OrderSide


class TestTriggerOrderExecution:
    limit_price = 100.0
    trigger_price = 150.0

    async def place_limit_order_with_trigger(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            price=self.limit_price,
            trigger_price=self.trigger_price,
        )

    async def place_market_order_with_trigger(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            trigger_price=self.trigger_price,
        )

    def simulate_filled_limit_order(self, sim, btc_usdt_perp: Symbol):
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp,
            low=self.limit_price - 10,
            high=self.limit_price + 20,
            close=self.limit_price + 15,
        )

    def simulate_filled_trigger_order(self, sim, btc_usdt_perp: Symbol):
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp,
            low=self.trigger_price - 10,
            high=self.trigger_price + 20,
            close=self.trigger_price + 15,
        )

    def simulate_filled_trigger_and_limit_orders(self, sim, btc_usdt_perp: Symbol):
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp,
            low=self.limit_price - 10,
            high=self.trigger_price + 20,
            close=self.trigger_price + 15,
        )

    def simulate_unfilled_trigger_order(self, sim, btc_usdt_perp: Symbol):
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp,
            low=self.trigger_price + 10,
            high=self.trigger_price + 20,
            close=self.trigger_price + 15,
        )

    async def test_limit_order_not_yet_triggered(self, sim, btc_usdt_perp: Symbol):
        await self.place_limit_order_with_trigger(sim, btc_usdt_perp)

        self.simulate_unfilled_trigger_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0
        assert sim.open_orders[0].kind == "trigger"

    async def test_triggered_limit_order_not_yet_executed(
        self, sim, btc_usdt_perp: Symbol
    ):
        await self.place_limit_order_with_trigger(sim, btc_usdt_perp)

        self.simulate_filled_trigger_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0
        assert sim.open_orders[0].kind == "limit"

    async def test_triggered_limit_order_executed_at_limit_price(
        self, sim, btc_usdt_perp: Symbol
    ):
        await self.place_limit_order_with_trigger(sim, btc_usdt_perp)

        self.simulate_filled_trigger_order(sim, btc_usdt_perp)
        self.simulate_filled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.open_positions[btc_usdt_perp].average_entry_price == self.limit_price

    async def test_triggered_limit_order_executed_on_same_frame(
        self, sim, btc_usdt_perp: Symbol
    ):
        await self.place_limit_order_with_trigger(sim, btc_usdt_perp)

        self.simulate_filled_trigger_and_limit_orders(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.open_positions[btc_usdt_perp].average_entry_price == self.limit_price

    async def test_market_order_not_yet_triggered(self, sim, btc_usdt_perp: Symbol):
        await self.place_market_order_with_trigger(sim, btc_usdt_perp)

        self.simulate_unfilled_trigger_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0
        assert sim.open_orders[0].kind == "trigger"

    async def test_triggered_market_order_executed_at_trigger_price(
        self, sim, btc_usdt_perp: Symbol
    ):
        await self.place_market_order_with_trigger(sim, btc_usdt_perp)

        self.simulate_filled_trigger_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert (
            sim.open_positions[btc_usdt_perp].average_entry_price == self.trigger_price
        )
