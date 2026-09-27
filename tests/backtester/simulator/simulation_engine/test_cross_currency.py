import math
from datetime import datetime

import pytest

from robottraderslab import Symbol
from robottraderslab._core import OHLCVRow, OHLCVsBySymbol, PlacementReserve
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FeeRates,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
    fee_model_for,
)
from robottraderslab.exchanges import OrderSide

AUD_CAD_PERP = Symbol.create("AUD/CAD:USD")
USD_CAD = Symbol("USD", "CAD")

INITIAL_USD = 10_000.0
ENTRY_PRICE = 0.90
USD_PER_CAD = 0.80


def make_tick(closes: dict[Symbol, float]) -> OHLCVsBySymbol:
    return {
        symbol: OHLCVRow(
            timestamp=datetime.now(),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1.0,
        )
        for symbol, close in closes.items()
    }


@pytest.fixture
def fill_recorder() -> CacheFillRecorder:
    return CacheFillRecorder()


@pytest.fixture
def engine(fill_recorder: CacheFillRecorder) -> FuturesSimulationEngine:
    return FuturesSimulationEngine(
        initial_balance={"USD": INITIAL_USD},
        fee_rates=FeeRates(maker=0.0, taker=0.0),
        fill_recorder=fill_recorder,
        margin_currency="USD",
        fee_model=fee_model_for("cost", PlacementReserve()),
    )


@pytest.fixture
def exchange(
    engine: FuturesSimulationEngine, fill_recorder: CacheFillRecorder
) -> SimulatedFuturesExchange:
    return SimulatedFuturesExchange(engine, fill_recorder)


async def open_long(
    engine: FuturesSimulationEngine,
    exchange: SimulatedFuturesExchange,
    quantity: float = 1000.0,
) -> None:
    tick = make_tick({AUD_CAD_PERP: ENTRY_PRICE, USD_CAD: 1 / USD_PER_CAD})
    engine.simulate_on_current_ohlcvs(datetime.now(), tick)
    await exchange.place_market_order(AUD_CAD_PERP, OrderSide.BUY, quantity)
    engine.simulate_on_current_ohlcvs(datetime.now(), tick)


class TestCrossCurrencyAccounting:
    async def test_entry_locks_margin_in_account_currency(self, engine, exchange):
        await open_long(engine, exchange)

        # 1000 AUD * 0.90 CAD * 0.80 USD/CAD at 1x leverage
        usd = engine.get_balances()["USD"]
        assert usd.locked == pytest.approx(720.0)
        assert usd.total == pytest.approx(INITIAL_USD)

    async def test_unrealised_pnl_converted_in_equity(self, engine, exchange):
        await open_long(engine, exchange)

        tick = make_tick({AUD_CAD_PERP: 0.95, USD_CAD: 1 / USD_PER_CAD})
        engine.simulate_on_current_ohlcvs(datetime.now(), tick)

        # +0.05 CAD on 1000 AUD = 50 CAD = 40 USD
        assert engine.get_equity("USD") == pytest.approx(INITIAL_USD + 40.0)

    async def test_exit_books_pnl_at_exit_rate(self, engine, exchange):
        await open_long(engine, exchange)

        exit_tick = make_tick({AUD_CAD_PERP: 0.95, USD_CAD: 1.0})
        engine.simulate_on_current_ohlcvs(datetime.now(), exit_tick)
        await exchange.place_market_order(
            AUD_CAD_PERP, OrderSide.SELL, 1000.0, reduce_only=True
        )

        # 50 CAD profit converted at the exit rate (1 CAD = 1 USD), while the
        # margin locked at the entry rate (720 USD) is fully released.
        usd = engine.get_balances()["USD"]
        assert usd.locked == pytest.approx(0.0)
        assert usd.total == pytest.approx(INITIAL_USD + 50.0)

    async def test_quote_conversion_rate_exposed_to_sizing(self, engine, exchange):
        tick = make_tick({AUD_CAD_PERP: ENTRY_PRICE, USD_CAD: 1 / USD_PER_CAD})
        engine.simulate_on_current_ohlcvs(datetime.now(), tick)

        assert await exchange.get_quote_conversion_rate(AUD_CAD_PERP) == pytest.approx(
            USD_PER_CAD
        )

    async def test_gap_candle_does_not_poison_conversion_rate(self, engine, exchange):
        await open_long(engine, exchange)

        gap_tick = make_tick({AUD_CAD_PERP: 0.95, USD_CAD: math.nan})
        engine.simulate_on_current_ohlcvs(datetime.now(), gap_tick)

        # The NaN close is ignored; the last real rate (0.80) still applies.
        assert engine.get_equity("USD") == pytest.approx(INITIAL_USD + 40.0)

    async def test_fills_record_conversion_rate(self, engine, exchange, fill_recorder):
        await open_long(engine, exchange)

        fills = fill_recorder.get_fills()
        assert fills.iloc[0]["rate"] == pytest.approx(USD_PER_CAD)
