import asyncio
import io
from collections.abc import Callable
from datetime import datetime, timezone

import pandas as pd
import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab.backtester.backtester import Backtester
from robottraderslab.backtester.simulator import SimulatedFuturesExchange
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.futures import FuturesAccount
from robottraderslab.ohlcv_provider import CSVOHLCVProvider
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    OHLCVs,
    StrategyProtocol,
    StrategyRequirements,
)

BTCUSDT_PERP = Symbol.create("BTC/USDT:USDT")

RISING_LOWS = """\
date,open,high,low,close,volume
2024-01-01T00:00:00+00:00,100,110,90,105,10
2024-01-02T00:00:00+00:00,105,115,95,110,10
2024-01-03T00:00:00+00:00,110,120,100,115,10
2024-01-04T00:00:00+00:00,115,125,105,120,10
"""

NEXT_CANDLE_DIPS = """\
date,open,high,low,close,volume
2024-01-01T00:00:00+00:00,100,110,94,105,10
2024-01-02T00:00:00+00:00,105,108,90,100,10
2024-01-03T00:00:00+00:00,100,104,96,102,10
"""

ENTRY_CANDLE_WICKS_DOWN = """\
date,open,high,low,close,volume
2024-01-01T00:00:00+00:00,100,110,80,105,10
2024-01-02T00:00:00+00:00,105,112,101,110,10
2024-01-03T00:00:00+00:00,110,111,85,95,10
"""

type BookingHandler = Callable[[int, OHLCVs, BookKeeper], None]


class _CSVStrategy(StrategyProtocol):
    market_type = "futures"

    def __init__(self, csv: str, handler: BookingHandler):
        self.ohlcv_provider = CSVOHLCVProvider(
            file=io.StringIO(csv), symbol=str(BTCUSDT_PERP), timeframe="1d"
        )
        self._handler = handler
        self._candle = 0

    async def setup(self, requirements: StrategyRequirements) -> None:
        requirements.ohlcv.add(BTCUSDT_PERP, "1d")

    def generate_trading_signals(self, ohlcvs: OHLCVs) -> None: ...

    def book_trading_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: set[TimeFrame],
    ) -> None:
        self._handler(self._candle, ohlcvs, bookkeeper)
        self._candle += 1


@pytest.fixture
def simulated_exchange() -> SimulatedFuturesExchange:
    return SimulatedFuturesExchange.create_from_settings(
        initial_balance={"USDT": 10_000.0},
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
    )


@pytest.fixture
def account(simulated_exchange: SimulatedFuturesExchange) -> FuturesAccount:
    return FuturesAccount(simulated_exchange)


@pytest.fixture
def run_backtest(
    simulated_exchange: SimulatedFuturesExchange,
) -> Callable[[str, BookingHandler], pd.DataFrame]:
    def runner(csv: str, handler: BookingHandler) -> pd.DataFrame:
        strategy = _CSVStrategy(csv, handler)
        backtester = Backtester(
            strategy,
            simulated_exchange.simulation_engine,
            strategy.ohlcv_provider,
            simulated_exchange.fill_recorder,
            "2024-01-01",
            "2024-01-05",
        )
        return asyncio.run(backtester.run()).fills

    return runner


def _moment(day: int) -> pd.Timestamp:
    return pd.Timestamp(datetime(2024, 1, day, tzinfo=timezone.utc))


def test_a_limit_at_the_candles_own_low_never_fills(account, run_backtest):
    def book_at_the_low(candle: int, ohlcvs: OHLCVs, bookkeeper: BookKeeper) -> None:
        low = ohlcvs.current(BTCUSDT_PERP, "1d", "low")
        bookkeeper.add(account.long_entry(BTCUSDT_PERP, 1.0).limit(low).build())

    fills = run_backtest(RISING_LOWS, book_at_the_low)

    assert fills.empty


def test_a_limit_the_next_candle_reaches_fills_on_that_candle(account, run_backtest):
    def book_once_inside_the_next_range(
        candle: int, ohlcvs: OHLCVs, bookkeeper: BookKeeper
    ) -> None:
        if candle == 0:
            bookkeeper.add(account.long_entry(BTCUSDT_PERP, 1.0).limit(95.0).build())

    fills = run_backtest(NEXT_CANDLE_DIPS, book_once_inside_the_next_range)

    assert list(fills.index) == [_moment(3)]
    assert list(fills["price"]) == [95.0]


def test_a_market_entry_fills_at_its_close_and_its_stop_waits_for_the_next_candle(
    account, run_backtest
):
    def enter_once_with_a_stop(
        candle: int, ohlcvs: OHLCVs, bookkeeper: BookKeeper
    ) -> None:
        if candle == 0:
            bookkeeper.add(
                account.long_entry(BTCUSDT_PERP, 1.0).stop_loss(90.0).build()
            )

    fills = run_backtest(ENTRY_CANDLE_WICKS_DOWN, enter_once_with_a_stop)

    assert list(fills["fill_type"]) == ["enter_long", "exit_long"]
    assert list(fills.index) == [_moment(2), _moment(4)]
    assert list(fills["price"]) == [105.0, 90.0]


def test_a_market_entry_with_its_stop_above_the_close_stops_the_run(
    account, run_backtest
):
    def enter_once_with_a_stop_above(
        candle: int, ohlcvs: OHLCVs, bookkeeper: BookKeeper
    ) -> None:
        if candle == 0:
            bookkeeper.add(
                account.long_entry(BTCUSDT_PERP, 1.0).stop_loss(120.0).build()
            )

    with pytest.raises(StrategyCriticalError, match="Invalid SL 120.0"):
        run_backtest(ENTRY_CANDLE_WICKS_DOWN, enter_once_with_a_stop_above)
