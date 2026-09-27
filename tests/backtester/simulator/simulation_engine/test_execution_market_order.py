import math

from robottraderslab import Symbol
from robottraderslab.exchanges import OrderSide, PositionSide


class TestMarketOrderExecution:
    quantity = 2.0
    close = 100.0

    async def place_enter_long(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=self.quantity
        )

    async def place_enter_short(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=self.quantity
        )

    async def place_exit_long(self, sim, btc_usdt_perp: Symbol, quantity: float = 2.0):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=quantity,
            reduce_only=True,
        )

    async def place_exit_short(self, sim, btc_usdt_perp: Symbol, quantity: float = 2.0):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=quantity,
            reduce_only=True,
        )

    def simulate_filled_market_order(self, sim, btc_usdt_perp: Symbol):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

    async def test_enter_long_executed_at_close_price(self, sim, btc_usdt_perp):
        await self.place_enter_long(sim, btc_usdt_perp)

        self.simulate_filled_market_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.open_positions[btc_usdt_perp].quantity == self.quantity
        assert sim.open_positions[btc_usdt_perp].average_entry_price == self.close
        assert sim.open_positions[btc_usdt_perp].side == PositionSide.LONG

    async def test_enter_short_executed_at_close_price(self, sim, btc_usdt_perp):
        await self.place_enter_short(sim, btc_usdt_perp)

        self.simulate_filled_market_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.open_positions[btc_usdt_perp].quantity == self.quantity
        assert sim.open_positions[btc_usdt_perp].average_entry_price == self.close
        assert sim.open_positions[btc_usdt_perp].side == PositionSide.SHORT

    async def test_exit_long_without_position(self, sim, btc_usdt_perp):
        await self.place_exit_long(sim, btc_usdt_perp)

        self.simulate_filled_market_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0

    async def test_exit_short_without_position(self, sim, btc_usdt_perp):
        await self.place_exit_short(sim, btc_usdt_perp)

        self.simulate_filled_market_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0

    async def test_exit_long_with_full_position(self, sim, btc_usdt_perp):
        await self.place_enter_long(sim, btc_usdt_perp)
        self.simulate_filled_market_order(sim, btc_usdt_perp)
        await self.place_exit_long(sim, btc_usdt_perp)

        self.simulate_filled_market_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0

    async def test_exit_short_with_full_position(self, sim, btc_usdt_perp):
        await self.place_enter_short(sim, btc_usdt_perp)
        self.simulate_filled_market_order(sim, btc_usdt_perp)
        await self.place_exit_short(sim, btc_usdt_perp)

        self.simulate_filled_market_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0

    async def test_increase_long_position(self, sim, btc_usdt_perp):
        await self.place_enter_long(sim, btc_usdt_perp)
        self.simulate_filled_market_order(sim, btc_usdt_perp)
        await self.place_enter_long(sim, btc_usdt_perp)

        self.simulate_filled_market_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.open_positions[btc_usdt_perp].quantity == self.quantity * 2
        assert sim.open_positions[btc_usdt_perp].side == PositionSide.LONG

    async def test_increase_short_position(self, sim, btc_usdt_perp):
        await self.place_enter_short(sim, btc_usdt_perp)
        self.simulate_filled_market_order(sim, btc_usdt_perp)
        await self.place_enter_short(sim, btc_usdt_perp)

        self.simulate_filled_market_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.open_positions[btc_usdt_perp].quantity == self.quantity * 2
        assert sim.open_positions[btc_usdt_perp].side == PositionSide.SHORT

    async def test_reduce_long_position(self, sim, btc_usdt_perp):
        await self.place_enter_long(sim, btc_usdt_perp)
        self.simulate_filled_market_order(sim, btc_usdt_perp)
        await self.place_exit_long(sim, btc_usdt_perp, self.quantity / 2)

        self.simulate_filled_market_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.open_positions[btc_usdt_perp].quantity == self.quantity / 2
        assert sim.open_positions[btc_usdt_perp].side == PositionSide.LONG

    async def test_reduce_short_position(self, sim, btc_usdt_perp):
        await self.place_enter_short(sim, btc_usdt_perp)
        self.simulate_filled_market_order(sim, btc_usdt_perp)
        await self.place_exit_short(sim, btc_usdt_perp, self.quantity / 2)

        self.simulate_filled_market_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.open_positions[btc_usdt_perp].quantity == self.quantity / 2
        assert sim.open_positions[btc_usdt_perp].side == PositionSide.SHORT


class TestMarketOrderOnAGapRow:
    quantity = 2.0
    close = 100.0

    async def test_placed_on_a_gap_row_fills_at_the_next_published_close(
        self, sim, btc_usdt_perp
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=math.nan)

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=self.quantity
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        assert sim.open_positions[btc_usdt_perp].average_entry_price == self.close

    async def test_resting_through_a_gap_row_fills_at_the_next_published_close(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=self.quantity
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=math.nan)
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        assert sim.open_positions[btc_usdt_perp].average_entry_price == self.close
