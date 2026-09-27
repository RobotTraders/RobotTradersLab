import pytest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import FeeRates
from robottraderslab.exchanges import OrderSide, PositionSide

FEE_RATE = 0.0005
QUANTITY = 10.0
ENTRY_PRICE = 100.0
EXIT_PRICE = 120.0


@pytest.fixture
def fee_rates() -> FeeRates:
    return FeeRates(maker=FEE_RATE, taker=FEE_RATE)


async def _enter_long(sim, btc_usdt_perp: Symbol) -> float:
    await sim.exchange.place_market_order(
        symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=QUANTITY
    )
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=ENTRY_PRICE)
    return sim.open_positions[btc_usdt_perp].quantity


async def _close_long(
    sim, btc_usdt_perp: Symbol, quantity: float, *, reduce_only: bool, price: float
) -> None:
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=price)
    await sim.exchange.place_market_order(
        symbol=btc_usdt_perp,
        side=OrderSide.SELL,
        quantity=quantity,
        reduce_only=reduce_only,
    )
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=price)


class TestClosePathsQuantityAndEquity:
    """An order sized to fully cover an existing position debits the same fee
    and produces the same equity whether it is reduce-only or not, and leaves
    no residual open either way."""

    async def test_reduce_only_close_leaves_no_residual(self, sim, btc_usdt_perp):
        position_quantity = await _enter_long(sim, btc_usdt_perp)

        await _close_long(
            sim, btc_usdt_perp, position_quantity, reduce_only=True, price=EXIT_PRICE
        )

        assert len(sim.open_positions) == 0

    async def test_opposing_entry_close_matches_hand_computed_equity(
        self, sim, btc_usdt_perp
    ):
        position_quantity = await _enter_long(sim, btc_usdt_perp)

        await _close_long(
            sim, btc_usdt_perp, position_quantity, reduce_only=False, price=EXIT_PRICE
        )

        fee_entry = QUANTITY * ENTRY_PRICE * FEE_RATE
        fee_exit = position_quantity * EXIT_PRICE * FEE_RATE
        realised_pnl = position_quantity * (EXIT_PRICE - ENTRY_PRICE)
        expected_equity = sim.initial_usdt - fee_entry - fee_exit + realised_pnl

        assert len(sim.open_positions) == 0
        assert sim.simulation_engine.get_equity("USDT") == pytest.approx(
            expected_equity
        )


class TestOpposingEntryMatchesVenueBehaviour:
    """A position closed by a non-reduce-only order of its exact quantity
    leaves the account flat."""

    async def test_opposing_entry_close_leaves_no_residual(self, sim, btc_usdt_perp):
        position_quantity = await _enter_long(sim, btc_usdt_perp)

        await _close_long(
            sim, btc_usdt_perp, position_quantity, reduce_only=False, price=ENTRY_PRICE
        )

        assert len(sim.open_positions) == 0


class TestOpposingEntryPartialClose:
    """A non-reduce-only order smaller than the position reduces it by
    exactly what was asked, unshaved."""

    async def test_partial_close_removes_the_requested_quantity_unshaved(
        self, sim, btc_usdt_perp
    ):
        position_quantity = await _enter_long(sim, btc_usdt_perp)
        partial_quantity = position_quantity / 2

        await _close_long(
            sim, btc_usdt_perp, partial_quantity, reduce_only=False, price=EXIT_PRICE
        )

        remaining = sim.open_positions[btc_usdt_perp]
        assert remaining.side == PositionSide.LONG
        assert remaining.quantity == pytest.approx(position_quantity - partial_quantity)


class TestOpposingEntryFlip:
    """An order larger than the existing position closes it in full."""

    async def test_flip_shaves_only_the_new_exposure(self, sim, btc_usdt_perp):
        position_quantity = await _enter_long(sim, btc_usdt_perp)
        flip_order_quantity = position_quantity + QUANTITY

        await _close_long(
            sim,
            btc_usdt_perp,
            flip_order_quantity,
            reduce_only=False,
            price=ENTRY_PRICE,
        )

        new_position = sim.open_positions[btc_usdt_perp]
        assert new_position.side == PositionSide.SHORT
        expected_new_quantity = (flip_order_quantity - position_quantity) * (
            1 - FEE_RATE
        )
        assert new_position.quantity == pytest.approx(expected_new_quantity)

    async def test_flip_reports_the_combined_filled_quantity(self, sim, btc_usdt_perp):
        position_quantity = await _enter_long(sim, btc_usdt_perp)
        flip_order_quantity = position_quantity + QUANTITY

        placed_order = await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=flip_order_quantity,
            reduce_only=False,
        )

        order_fill = await placed_order.get_fill()
        new_position = sim.open_positions[btc_usdt_perp]
        assert order_fill.quantity == pytest.approx(
            position_quantity + new_position.quantity
        )
