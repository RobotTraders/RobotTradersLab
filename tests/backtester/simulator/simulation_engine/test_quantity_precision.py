import pytest

from robottraderslab.backtester.simulator import FeeRates
from robottraderslab.exchanges import OrderSide


@pytest.fixture
def fee_rates() -> FeeRates:
    return FeeRates(maker=0.0002, taker=0.0005)


class TestQuantitiesAVenueCouldReport:
    """A venue quantises what it fills, so a quantity it reports can be sent
    back to it and mean the same amount. The simulator answers the same
    orders, so what it fills has to survive the same round trip.
    """

    close = 2500.0

    async def test_closing_two_fills_in_turn_leaves_no_position(
        self, sim, eth_usdt_perp
    ):
        first = await self._fill(sim, eth_usdt_perp, OrderSide.BUY, 0.03)
        second = await self._fill(sim, eth_usdt_perp, OrderSide.BUY, 0.05)

        await self._fill(sim, eth_usdt_perp, OrderSide.SELL, second)
        await self._fill(sim, eth_usdt_perp, OrderSide.SELL, first)

        assert sim.open_positions == {}

    async def test_a_fill_is_reported_at_a_tradable_quantity(self, sim, eth_usdt_perp):
        filled = await self._fill(sim, eth_usdt_perp, OrderSide.BUY, 0.03)

        assert filled == pytest.approx(round(filled, 8), abs=0.0)

    async def _fill(self, sim, symbol, side, quantity) -> float:
        sim.simulate_on_current_ohlcvs(symbol, close=self.close)
        order = await sim.exchange.place_market_order(
            symbol=symbol, side=side, quantity=quantity
        )
        fill = await order.get_fill()
        return fill.quantity
