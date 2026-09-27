import pytest

from robottraderslab.backtester.simulator import FeeRates
from robottraderslab.exceptions import ExchangeRecoverableError
from robottraderslab.exchanges import OrderSide, PositionSide

FEE_RATE = 0.0005
PRICE = 2500.0
ORDERED = 0.05
OPENED = 0.049975
REDUCED_BY = 0.019975
LEFT_OPEN = 0.03
FLIP_ORDERED = 0.09995
FLIPPED_TO = 0.04995001
FLIP_FILLED = 0.09992501
UNDER_ONE_STEP = 0.000000004
ORDERED_OFF_STEP = 0.0500000004
FEE_ON_THE_STEPS_ORDERED = 0.0625
INITIAL_USDT = 10_000.0


@pytest.fixture
def fee_rates() -> FeeRates:
    return FeeRates(maker=FEE_RATE, taker=FEE_RATE)


async def _enter_long(sim, symbol) -> None:
    await sim.exchange.place_market_order(
        symbol=symbol, side=OrderSide.BUY, quantity=ORDERED
    )
    sim.simulate_on_current_ohlcvs(symbol, close=PRICE)


class TestEveryPathMovesWholeSteps:
    """A strategy can send a reported quantity straight back and mean the same amount."""

    async def test_an_entry_opens_a_position_of_whole_steps(self, sim, eth_usdt_perp):
        await _enter_long(sim, eth_usdt_perp)

        assert sim.open_positions[eth_usdt_perp].quantity == OPENED

    async def test_an_entry_reports_the_quantity_it_opened(self, sim, eth_usdt_perp):
        placed_order = await sim.exchange.place_market_order(
            symbol=eth_usdt_perp, side=OrderSide.BUY, quantity=ORDERED
        )
        sim.simulate_on_current_ohlcvs(eth_usdt_perp, close=PRICE)

        fill = await placed_order.get_fill()

        assert fill.quantity == OPENED

    async def test_a_reduce_only_exit_leaves_whole_steps_open(self, sim, eth_usdt_perp):
        await _enter_long(sim, eth_usdt_perp)

        await sim.exchange.place_market_order(
            symbol=eth_usdt_perp,
            side=OrderSide.SELL,
            quantity=REDUCED_BY,
            reduce_only=True,
        )
        sim.simulate_on_current_ohlcvs(eth_usdt_perp, close=PRICE)

        assert sim.open_positions[eth_usdt_perp].quantity == LEFT_OPEN

    async def test_a_reduce_only_exit_reports_what_it_removed(self, sim, eth_usdt_perp):
        await _enter_long(sim, eth_usdt_perp)

        placed_order = await sim.exchange.place_market_order(
            symbol=eth_usdt_perp,
            side=OrderSide.SELL,
            quantity=REDUCED_BY,
            reduce_only=True,
        )

        fill = await placed_order.get_fill()

        assert fill.quantity == REDUCED_BY

    async def test_an_order_past_the_position_flips_onto_whole_steps(
        self, sim, eth_usdt_perp
    ):
        await _enter_long(sim, eth_usdt_perp)

        await sim.exchange.place_market_order(
            symbol=eth_usdt_perp, side=OrderSide.SELL, quantity=FLIP_ORDERED
        )
        sim.simulate_on_current_ohlcvs(eth_usdt_perp, close=PRICE)

        flipped = sim.open_positions[eth_usdt_perp]
        assert flipped.side == PositionSide.SHORT
        assert flipped.quantity == FLIPPED_TO

    async def test_an_order_past_the_position_reports_both_legs(
        self, sim, eth_usdt_perp
    ):
        await _enter_long(sim, eth_usdt_perp)

        placed_order = await sim.exchange.place_market_order(
            symbol=eth_usdt_perp, side=OrderSide.SELL, quantity=FLIP_ORDERED
        )

        fill = await placed_order.get_fill()

        assert fill.quantity == FLIP_FILLED

    async def test_a_liquidation_records_the_position_in_whole_steps(
        self, sim, eth_usdt_perp
    ):
        sim.simulation_engine.set_symbol_leverage(eth_usdt_perp, 5.0)
        await _enter_long(sim, eth_usdt_perp)

        sim.simulate_on_current_ohlcvs(
            eth_usdt_perp, low=1_900.0, high=2_100.0, close=1_950.0
        )

        fills = sim.fill_recorder.get_fills()
        liquidation = fills[fills["fill_type"] == "liquidate_long"].iloc[0]
        assert liquidation["net_quantity"] == OPENED

    async def test_a_fee_is_charged_on_the_steps_the_order_bought(
        self, sim, eth_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=eth_usdt_perp, side=OrderSide.BUY, quantity=ORDERED_OFF_STEP
        )
        sim.simulate_on_current_ohlcvs(eth_usdt_perp, close=PRICE)

        balances = await sim.exchange.get_balances([eth_usdt_perp])

        assert balances["USDT"].total == INITIAL_USDT - FEE_ON_THE_STEPS_ORDERED

    async def test_an_order_under_one_step_is_refused_at_placement(
        self, sim, eth_usdt_perp
    ):
        with pytest.raises(ExchangeRecoverableError):
            await sim.exchange.place_market_order(
                symbol=eth_usdt_perp, side=OrderSide.BUY, quantity=UNDER_ONE_STEP
            )
