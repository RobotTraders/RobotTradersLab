from robottraderslab import Symbol
from robottraderslab.exchanges import OrderSide
from robottraderslab.futures.futures_close_position import ClosePositionAction

HELD = 2.0
CLOSE = 100.0


async def _hold_long(sim, btc_usdt_perp: Symbol) -> None:
    await sim.exchange.place_market_order(
        symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=HELD
    )
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=CLOSE)


async def test_closes_the_whole_position_the_simulator_holds(sim, btc_usdt_perp):
    await _hold_long(sim, btc_usdt_perp)
    close = ClosePositionAction(exchange=sim.exchange, symbol=btc_usdt_perp)

    await close.execute()
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=CLOSE)

    assert sim.open_positions == {}


async def test_a_closing_ratio_leaves_the_rest_of_the_position(sim, btc_usdt_perp):
    await _hold_long(sim, btc_usdt_perp)
    close = ClosePositionAction(
        exchange=sim.exchange, symbol=btc_usdt_perp, closing_ratio=0.5
    )

    await close.execute()
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=CLOSE)

    assert sim.open_positions[btc_usdt_perp].quantity == 1.0


async def test_a_limit_close_rests_reduce_only_until_price_reaches_it(
    sim, btc_usdt_perp
):
    await _hold_long(sim, btc_usdt_perp)
    close = ClosePositionAction(
        exchange=sim.exchange, symbol=btc_usdt_perp, limit_price=120.0
    )

    await close.execute()
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=110.0)
    resting = sim.open_orders[0]
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=120.0)

    assert resting.limit_price == 120.0
    assert resting.reduce_only is True
    assert sim.open_positions == {}


async def test_a_flat_simulator_is_left_flat(sim, btc_usdt_perp):
    close = ClosePositionAction(exchange=sim.exchange, symbol=btc_usdt_perp)

    closed = await close.execute()

    assert closed.orders == ()
    assert sim.open_orders == []
