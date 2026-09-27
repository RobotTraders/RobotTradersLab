import pytest

from robottraderslab import Symbol
from robottraderslab._core import PlacementReserve
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FeeModel,
    FeeRates,
    fee_model_for,
)
from robottraderslab.exceptions import ExchangeRecoverableError
from robottraderslab.exchanges import OrderSide, PositionSide

FEE_RATE = 0.0005
_NO_RESERVE = PlacementReserve(margin_markup=0.0, fee_markup=0.0, notional_reserve=0.0)
QUANTITY = 10.0
ENTRY_PRICE = 100.0
EXIT_PRICE = 120.0
ENTRY_FEE = QUANTITY * ENTRY_PRICE * FEE_RATE
EXIT_FEE = QUANTITY * EXIT_PRICE * FEE_RATE
MARGIN = QUANTITY * ENTRY_PRICE
FLIP_QUANTITY = 15.0
UNDER_ONE_STEP_AFTER_THE_CUT = 0.7e-8
KNIFE_EDGE_PRICE = 42708.7
_FEE_COLUMN = CacheFillRecorder.HEADER.index("fee")


@pytest.fixture
def fee_rates() -> FeeRates:
    return FeeRates(maker=FEE_RATE, taker=FEE_RATE)


@pytest.fixture
def fee_model() -> FeeModel:
    return fee_model_for("exchange", _NO_RESERVE)


async def _enter_long(sim, btc_usdt_perp: Symbol) -> None:
    await sim.exchange.place_market_order(
        symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=QUANTITY
    )
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=ENTRY_PRICE)


async def _flip_short(sim, btc_usdt_perp: Symbol, quantity: float) -> None:
    await sim.exchange.place_market_order(
        symbol=btc_usdt_perp,
        side=OrderSide.SELL,
        quantity=quantity,
        reduce_only=False,
    )
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=ENTRY_PRICE)


async def _exit_long(sim, btc_usdt_perp: Symbol) -> None:
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=EXIT_PRICE)
    await sim.exchange.place_market_order(
        symbol=btc_usdt_perp,
        side=OrderSide.SELL,
        quantity=sim.open_positions[btc_usdt_perp].quantity,
        reduce_only=True,
    )
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=EXIT_PRICE)


class TestEntry:
    """A venue fills the quantity ordered and takes the fee out of the margin
    currency, so the position is never shaved by what it paid.
    """

    async def test_position_holds_the_quantity_ordered(self, sim, btc_usdt_perp):
        await _enter_long(sim, btc_usdt_perp)

        assert sim.open_positions[btc_usdt_perp].quantity == pytest.approx(QUANTITY)

    async def test_balance_pays_the_fee_beside_the_full_margin(
        self, sim, btc_usdt_perp
    ):
        await _enter_long(sim, btc_usdt_perp)

        balance = sim.simulation_engine.get_balances()["USDT"]
        assert balance.total == pytest.approx(sim.initial_usdt - ENTRY_FEE)
        assert balance.locked == pytest.approx(MARGIN)


class TestRoundTrip:
    """The exit realises the whole quantity the entry opened, so the round trip
    leaves the balance short of its two fees alone.
    """

    async def test_balance_keeps_the_profit_net_of_both_fees(self, sim, btc_usdt_perp):
        await _enter_long(sim, btc_usdt_perp)

        await _exit_long(sim, btc_usdt_perp)

        realised = QUANTITY * (EXIT_PRICE - ENTRY_PRICE)
        assert sim.simulation_engine.get_equity("USDT") == pytest.approx(
            sim.initial_usdt + realised - ENTRY_FEE - EXIT_FEE
        )

    async def test_exit_leaves_no_position(self, sim, btc_usdt_perp):
        await _enter_long(sim, btc_usdt_perp)

        await _exit_long(sim, btc_usdt_perp)

        assert sim.open_positions == {}


class TestFlip:
    """An order that closes a position and opens the other side executes two
    quantities, and each leg is charged on the one it executed.
    """

    async def test_each_leg_pays_on_the_quantity_it_executed(self, sim, btc_usdt_perp):
        await _enter_long(sim, btc_usdt_perp)

        await _flip_short(sim, btc_usdt_perp, FLIP_QUANTITY)

        close_fee, remainder_fee = (
            fill[_FEE_COLUMN] for fill in sim.fill_recorder.fills[-2:]
        )
        assert close_fee == pytest.approx(QUANTITY * ENTRY_PRICE * FEE_RATE)
        assert remainder_fee == pytest.approx(
            (FLIP_QUANTITY - QUANTITY) * ENTRY_PRICE * FEE_RATE
        )

    async def test_the_flipped_side_holds_the_excess_unshaved(self, sim, btc_usdt_perp):
        await _enter_long(sim, btc_usdt_perp)

        await _flip_short(sim, btc_usdt_perp, FLIP_QUANTITY)

        flipped = sim.open_positions[btc_usdt_perp]
        assert flipped.side == PositionSide.SHORT
        assert flipped.quantity == pytest.approx(FLIP_QUANTITY - QUANTITY)

    async def test_balance_pays_the_whole_order_fee_once(self, sim, btc_usdt_perp):
        await _enter_long(sim, btc_usdt_perp)

        await _flip_short(sim, btc_usdt_perp, FLIP_QUANTITY)

        balance = sim.simulation_engine.get_balances()["USDT"]
        assert balance.total == pytest.approx(
            sim.initial_usdt - ENTRY_FEE - FLIP_QUANTITY * ENTRY_PRICE * FEE_RATE
        )
        assert balance.locked == pytest.approx((FLIP_QUANTITY - QUANTITY) * ENTRY_PRICE)


class TestAnOrderSizedToTheWholeBalance:
    """A quantity sized to the last unit of the balance fits only if nothing
    rounds it up, and the simulator cuts an order down to its step the way a
    venue cuts to its lot.
    """

    async def test_fills_at_a_price_whose_exact_quantity_rounds_up(
        self, sim, btc_usdt_perp
    ):
        quantity = sim.initial_usdt / (KNIFE_EDGE_PRICE * (1 + FEE_RATE))

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=quantity
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=KNIFE_EDGE_PRICE)

        assert sim.open_positions[btc_usdt_perp].quantity <= quantity
        assert sim.open_positions[btc_usdt_perp].quantity == pytest.approx(quantity)


class TestAnOrderUnderOneStepAfterTheCut:
    """The placement guard counts an order the way the fill counts it, so a
    quantity that rounds to a step but cuts to none never reaches the engine.
    """

    async def test_is_refused_at_placement(self, sim, btc_usdt_perp):
        with pytest.raises(ExchangeRecoverableError):
            await sim.exchange.place_market_order(
                symbol=btc_usdt_perp,
                side=OrderSide.BUY,
                quantity=UNDER_ONE_STEP_AFTER_THE_CUT,
            )
