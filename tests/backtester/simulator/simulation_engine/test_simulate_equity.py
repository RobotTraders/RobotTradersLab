from robottraderslab import Symbol
from robottraderslab.exchanges import OrderSide


class TestEquity:
    async def test_initial_equity(self, sim):
        assert sim.simulation_engine.get_equity("USDT") == 10_000.0

    async def test_after_ordering(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(btc_usdt_perp, OrderSide.BUY, 1.0)

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0
        assert sim.simulation_engine.get_equity("USDT") == 10_000.0

    async def test_after_taking_position(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(btc_usdt_perp, OrderSide.BUY, 1.0)
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=20.0)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.simulation_engine.get_equity("USDT") == 10_000.0

    async def test_position_price_is_higher(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(btc_usdt_perp, OrderSide.BUY, 1.0)
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=20.0)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=40.0)

        assert sim.simulation_engine.get_equity("USDT") == 10_020.0

    async def test_position_price_is_lower(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(btc_usdt_perp, OrderSide.BUY, 1.0)
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=20.0)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=10.0)

        assert sim.simulation_engine.get_equity("USDT") == 9990.0
