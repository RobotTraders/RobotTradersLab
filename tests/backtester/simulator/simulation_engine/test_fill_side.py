from robottraderslab import Symbol
from robottraderslab.exchanges import OrderSide, PositionSide


class TestFuturesFillSide:
    quantity = 1.0
    close = 100.0

    async def test_enter_long_records_long_side(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=self.quantity
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        fill = sim.fill_recorder.fills[0]
        assert fill[2] == PositionSide.LONG

    async def test_enter_short_records_short_side(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=self.quantity
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        fill = sim.fill_recorder.fills[0]
        assert fill[2] == PositionSide.SHORT
