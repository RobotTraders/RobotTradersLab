from unittest.mock import AsyncMock

import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import ExchangeRecoverableError, NoOpenPositionError
from robottraderslab.exchanges import OrderSide
from robottraderslab.futures import close_held_position

HELD = 2.0
CLOSE = 100.0


@pytest.fixture
async def held_long(sim, btc_usdt_perp: Symbol) -> None:
    await sim.exchange.place_market_order(
        symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=HELD
    )
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=CLOSE)


@pytest.fixture
async def held_short(sim, btc_usdt_perp: Symbol) -> None:
    await sim.exchange.place_market_order(
        symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=HELD
    )
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=CLOSE)


@pytest.fixture
def counted_exchange(sim) -> AsyncMock:
    """The simulator's exchange, every call to it recorded and passed on."""
    return AsyncMock(wraps=sim.exchange)


async def test_closes_the_whole_position_with_one_read_and_one_order(
    held_long, sim, btc_usdt_perp, counted_exchange
):
    await close_held_position(counted_exchange, btc_usdt_perp)
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=CLOSE)

    assert sim.open_positions == {}
    assert counted_exchange.get_open_positions.await_count == 1
    counted_exchange.place_market_order.assert_awaited_once_with(
        btc_usdt_perp, OrderSide.SELL, HELD, reduce_only=True
    )


async def test_closes_a_quantity_with_one_read_and_one_order(
    held_long, sim, btc_usdt_perp, counted_exchange
):
    await close_held_position(counted_exchange, btc_usdt_perp, 0.5)
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=CLOSE)

    assert sim.open_positions[btc_usdt_perp].quantity == HELD - 0.5
    assert counted_exchange.get_open_positions.await_count == 1
    assert counted_exchange.place_market_order.await_count == 1


async def test_a_short_closes_with_a_buy(held_short, btc_usdt_perp, counted_exchange):
    await close_held_position(counted_exchange, btc_usdt_perp)

    counted_exchange.place_market_order.assert_awaited_once_with(
        btc_usdt_perp, OrderSide.BUY, HELD, reduce_only=True
    )


async def test_a_symbol_holding_no_position_is_refused_with_no_order(
    btc_usdt_perp, counted_exchange
):
    with pytest.raises(NoOpenPositionError, match="BTC/USDT:USDT"):
        await close_held_position(counted_exchange, btc_usdt_perp)

    assert counted_exchange.place_market_order.await_count == 0


async def test_a_quantity_above_the_position_is_refused_with_no_order(
    held_long, btc_usdt_perp, counted_exchange
):
    with pytest.raises(ExchangeRecoverableError, match="at most the position's"):
        await close_held_position(counted_exchange, btc_usdt_perp, HELD + 1)

    assert counted_exchange.place_market_order.await_count == 0


@pytest.mark.parametrize("quantity", [0.0, -1.0])
async def test_a_quantity_not_above_zero_is_refused_with_no_order(
    held_long, btc_usdt_perp, counted_exchange, quantity
):
    with pytest.raises(ValueError, match="greater than 0"):
        await close_held_position(counted_exchange, btc_usdt_perp, quantity)

    assert counted_exchange.place_market_order.await_count == 0
