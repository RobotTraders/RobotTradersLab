import pytest

from robottraderslab.exchanges import OrderSide, PositionSide


class TestPositionNetting:
    """One-way mode netting: opposing entry orders net against existing positions."""

    close = 100.0

    async def test_partial_close_when_opposing_entry_smaller(self, sim, btc_usdt_perp):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=10.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=3.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        assert sim.open_positions[btc_usdt_perp].side == PositionSide.LONG
        assert sim.open_positions[btc_usdt_perp].quantity == pytest.approx(7.0)

    async def test_full_close_when_equal_opposing_entry(self, sim, btc_usdt_perp):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=10.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=10.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        assert len(sim.open_positions) == 0

    async def test_flip_when_opposing_entry_larger(self, sim, btc_usdt_perp):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=10.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=15.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        assert sim.open_positions[btc_usdt_perp].side == PositionSide.SHORT
        assert sim.open_positions[btc_usdt_perp].quantity == pytest.approx(5.0)

    async def test_partial_close_short_with_buy(self, sim, btc_usdt_perp):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=10.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=3.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        assert sim.open_positions[btc_usdt_perp].side == PositionSide.SHORT
        assert sim.open_positions[btc_usdt_perp].quantity == pytest.approx(7.0)

    async def test_flip_short_to_long(self, sim, btc_usdt_perp):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=10.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=15.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        assert sim.open_positions[btc_usdt_perp].side == PositionSide.LONG
        assert sim.open_positions[btc_usdt_perp].quantity == pytest.approx(5.0)

    async def test_same_side_entry_adds_to_position(self, sim, btc_usdt_perp):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=10.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=5.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        assert sim.open_positions[btc_usdt_perp].side == PositionSide.LONG
        assert sim.open_positions[btc_usdt_perp].quantity == pytest.approx(15.0)

    async def test_reduce_only_exit_without_position(self, sim, btc_usdt_perp):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=1.0,
            reduce_only=True,
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.close)

        assert len(sim.open_positions) == 0

    async def test_full_close_with_profit(self, sim, btc_usdt_perp):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=1.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        balance_after_entry = sim.simulation_engine.get_balances()["USDT"]

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=120.0)
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=1.0
        )

        balance_after_close = sim.simulation_engine.get_balances()["USDT"]
        assert balance_after_close.total > balance_after_entry.total

    async def test_full_close_with_loss(self, sim, btc_usdt_perp):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=1.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        balance_after_entry = sim.simulation_engine.get_balances()["USDT"]

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=80.0)
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=1.0
        )

        balance_after_close = sim.simulation_engine.get_balances()["USDT"]
        assert balance_after_close.total < balance_after_entry.total
