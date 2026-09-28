from datetime import UTC, datetime

from robottraderslab import Symbol
from robottraderslab._core import OHLCVRow
from robottraderslab.exchanges import OrderSide, StopLoss, TakeProfit

ENTRY_PRICE = 100.0
FIRST_CLOSE = datetime(2026, 3, 1, 15, 15, tzinfo=UTC)
SECOND_CLOSE = datetime(2026, 3, 1, 15, 30, tzinfo=UTC)
THIRD_CLOSE = datetime(2026, 3, 1, 15, 45, tzinfo=UTC)


def _play_candle(
    sim, symbol: Symbol, close_moment: datetime, low: float, high: float
) -> None:
    candle = OHLCVRow(
        timestamp=close_moment,
        open=ENTRY_PRICE,
        high=high,
        low=low,
        close=ENTRY_PRICE,
        volume=1.0,
    )
    sim.simulation_engine.simulate_on_current_ohlcvs(close_moment, {symbol: candle})


async def test_take_profit_fills_on_the_first_later_candle_reaching_it(
    sim, btc_usdt_perp
):
    await sim.exchange.place_limit_order(
        symbol=btc_usdt_perp,
        side=OrderSide.SELL,
        quantity=1.0,
        price=ENTRY_PRICE,
        take_profit=TakeProfit(trigger_price=99.0),
    )

    _play_candle(sim, btc_usdt_perp, FIRST_CLOSE, low=98.5, high=100.5)
    _play_candle(sim, btc_usdt_perp, SECOND_CLOSE, low=100.0, high=101.0)
    _play_candle(sim, btc_usdt_perp, THIRD_CLOSE, low=98.0, high=100.0)

    fills = sim.fill_recorder.get_fills()
    assert list(fills.index) == [FIRST_CLOSE, THIRD_CLOSE]
    assert list(fills["price"]) == [ENTRY_PRICE, 99.0]


async def test_stop_loss_rests_through_the_candle_its_entry_fills_on(
    sim, btc_usdt_perp
):
    await sim.exchange.place_limit_order(
        symbol=btc_usdt_perp,
        side=OrderSide.SELL,
        quantity=1.0,
        price=ENTRY_PRICE,
        stop_loss=StopLoss(trigger_price=101.0),
    )

    _play_candle(sim, btc_usdt_perp, FIRST_CLOSE, low=99.0, high=101.5)
    _play_candle(sim, btc_usdt_perp, SECOND_CLOSE, low=99.0, high=100.0)

    assert sim.open_positions[btc_usdt_perp].quantity == 1.0
    assert [order.kind for order in sim.open_orders] == ["stop-loss"]
