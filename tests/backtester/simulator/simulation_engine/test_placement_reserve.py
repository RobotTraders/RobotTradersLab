import logging

import pytest

from robottraderslab import Symbol
from robottraderslab._core import PlacementReserve
from robottraderslab.backtester.simulator import FeeModel, FeeRates, fee_model_for
from robottraderslab.exceptions import ExchangeRecoverableError
from robottraderslab.exchanges import OrderSide, PositionSide

TAKER_FEE_RATE = 0.0006
LEVERAGE = 10.0
PRICE = 100.0
BITGET_RESERVE = PlacementReserve(
    margin_markup=0.01, fee_markup=0.005, notional_reserve=0.003
)
ABOVE_THE_BOUNDARY = 0.96
BELOW_THE_BOUNDARY = 0.95
HELD_QUANTITY = 100.0
FLIP_QUANTITY = 1050.0
OVERSIZED_FLIP_QUANTITY = 1060.0


@pytest.fixture
def fee_rates() -> FeeRates:
    return FeeRates(maker=TAKER_FEE_RATE, taker=TAKER_FEE_RATE)


@pytest.fixture
def fee_model() -> FeeModel:
    return fee_model_for("exchange", BITGET_RESERVE)


@pytest.fixture
def levered_sim(sim, btc_usdt_perp: Symbol):
    """Bitget's markups put a boundary between the two shares this fixture
    straddles.
    """
    sim.simulation_engine.set_symbol_leverage(btc_usdt_perp, LEVERAGE)
    return sim


def _quantity_at(share: float, sim) -> float:
    return share * sim.initial_usdt * LEVERAGE / PRICE


async def _fill_long(sim, symbol: Symbol, quantity: float) -> None:
    await sim.exchange.place_market_order(
        symbol=symbol, side=OrderSide.BUY, quantity=quantity
    )
    sim.simulate_on_current_ohlcvs(symbol, close=PRICE)


class TestMarketOrder:
    """A market order is weighed when it fills, so an order over the boundary
    opens nothing and leaves the reason in the log.
    """

    async def test_order_above_the_boundary_opens_no_position(
        self, levered_sim, btc_usdt_perp, caplog
    ):
        await levered_sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=_quantity_at(ABOVE_THE_BOUNDARY, levered_sim),
        )

        with caplog.at_level(logging.WARNING):
            levered_sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=PRICE)

        assert levered_sim.open_positions == {}
        assert "requires" in caplog.text

    async def test_order_below_the_boundary_fills(self, levered_sim, btc_usdt_perp):
        quantity = _quantity_at(BELOW_THE_BOUNDARY, levered_sim)

        await levered_sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=quantity
        )
        levered_sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=PRICE)

        assert levered_sim.open_positions[btc_usdt_perp].quantity == pytest.approx(
            quantity
        )


class TestLimitOrder:
    """A venue weighs a limit entry when it is placed, whatever its side, so
    one over the boundary never reaches the book.
    """

    @pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
    async def test_order_above_the_boundary_is_refused_at_placement(
        self, levered_sim, btc_usdt_perp, side
    ):
        with pytest.raises(ExchangeRecoverableError):
            await levered_sim.exchange.place_limit_order(
                symbol=btc_usdt_perp,
                side=side,
                quantity=_quantity_at(ABOVE_THE_BOUNDARY, levered_sim),
                price=PRICE,
            )

        assert levered_sim.open_orders == []

    @pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
    async def test_order_below_the_boundary_rests(
        self, levered_sim, btc_usdt_perp, side
    ):
        await levered_sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=side,
            quantity=_quantity_at(BELOW_THE_BOUNDARY, levered_sim),
            price=PRICE,
        )

        assert len(levered_sim.open_orders) == 1

    async def test_an_order_opposing_the_position_locks_what_its_close_leaves_unmet(
        self, levered_sim, btc_usdt_perp
    ):
        await _fill_long(levered_sim, btc_usdt_perp, HELD_QUANTITY)
        locked_by_the_position = levered_sim.simulation_engine.get_balances()[
            "USDT"
        ].locked

        await levered_sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=FLIP_QUANTITY,
            price=PRICE,
        )

        locked = levered_sim.simulation_engine.get_balances()["USDT"].locked
        opening_margin = (FLIP_QUANTITY - HELD_QUANTITY) * PRICE / LEVERAGE
        assert locked - locked_by_the_position == pytest.approx(
            opening_margin - locked_by_the_position
        )

    async def test_an_order_closing_the_position_rests_with_nothing_available(
        self, levered_sim, btc_usdt_perp
    ):
        quantity = _quantity_at(BELOW_THE_BOUNDARY, levered_sim)
        await _fill_long(levered_sim, btc_usdt_perp, quantity)
        locked_by_the_position = levered_sim.simulation_engine.get_balances()[
            "USDT"
        ].locked

        await levered_sim.exchange.place_limit_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=quantity, price=PRICE
        )

        assert len(levered_sim.open_orders) == 1
        locked = levered_sim.simulation_engine.get_balances()["USDT"].locked
        assert locked == pytest.approx(locked_by_the_position)


class TestAnOrderOpposingThePosition:
    """A venue verifies the value an order opens against the balance plus the
    margin its closing leg releases, so a close is never weighed and a flip is
    weighed on its excess with the closed position's margin counted in. Holding
    100 units at leverage 10 leaves 8,994 available and 1,000 backing the
    position, which together carry an opening leg of 955 units and no more.
    """

    async def test_a_close_needs_nothing_available(self, levered_sim, btc_usdt_perp):
        quantity = _quantity_at(BELOW_THE_BOUNDARY, levered_sim)
        await _fill_long(levered_sim, btc_usdt_perp, quantity)

        await levered_sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=quantity,
            reduce_only=False,
        )
        levered_sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=PRICE)

        assert levered_sim.open_positions == {}

    async def test_a_flip_counts_the_margin_its_close_releases(
        self, levered_sim, btc_usdt_perp
    ):
        await _fill_long(levered_sim, btc_usdt_perp, HELD_QUANTITY)

        await levered_sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=FLIP_QUANTITY,
            reduce_only=False,
        )
        levered_sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=PRICE)

        flipped = levered_sim.open_positions[btc_usdt_perp]
        assert flipped.side == PositionSide.SHORT
        assert flipped.quantity == pytest.approx(FLIP_QUANTITY - HELD_QUANTITY)

    async def test_a_flip_beyond_the_balance_and_the_released_margin_keeps_the_position(
        self, levered_sim, btc_usdt_perp
    ):
        await _fill_long(levered_sim, btc_usdt_perp, HELD_QUANTITY)

        with pytest.raises(ExchangeRecoverableError, match="requires"):
            await levered_sim.exchange.place_market_order(
                symbol=btc_usdt_perp,
                side=OrderSide.SELL,
                quantity=OVERSIZED_FLIP_QUANTITY,
                reduce_only=False,
            )

        held = levered_sim.open_positions[btc_usdt_perp]
        assert held.side == PositionSide.LONG
        assert held.quantity == pytest.approx(HELD_QUANTITY)
