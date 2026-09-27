import asyncio
import io
from collections.abc import Callable
from datetime import datetime

import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab.backtester.backtester import Backtester
from robottraderslab.backtester.simulator import CacheFillRecorder
from robottraderslab.ohlcv_provider import CSVOHLCVProvider
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    OHLCVs,
    StrategyProtocol,
    StrategyRequirements,
)


@pytest.fixture
def csv_3_daily_no_price_fluctuation():
    return io.StringIO("""\
date,open,high,low,close,volume
2024-01-01T00:00:00+00:00,100,100,100,100,10
2024-01-02T00:00:00+00:00,100,100,100,100,10
2024-01-03T00:00:00+00:00,100,100,100,100,10
""")


@pytest.fixture
def csv_3_daily_with_price_increases():
    return io.StringIO("""\
date,open,high,low,close,volume
2024-01-01T00:00:00+00:00,100,110,100,110,10
2024-01-02T00:00:00+00:00,110,120,110,120,10
2024-01-03T00:00:00+00:00,120,130,120,130,10
""")


@pytest.fixture
def csv_3_daily_with_price_decreases():
    return io.StringIO("""\
date,open,high,low,close,volume
2024-01-01T00:00:00+00:00,100,110,90,90,10
2024-01-02T00:00:00+00:00,90,100,80,80,10
2024-01-03T00:00:00+00:00,80,90,70,70,10
""")


class CSVOHLCVStrategyFixture:
    """Base fixture that implements load and generate_trading_signals."""

    def __init__(self, csv: io.StringIO, symbol: Symbol):
        self._symbol = symbol
        self.ohlcv_provider = CSVOHLCVProvider(
            file=csv,
            symbol=str(symbol),
            timeframe="1d",
        )

    def generate_trading_signals(self, ohlcvs: OHLCVs) -> None: ...

    async def setup(self, requirements: StrategyRequirements) -> None:
        requirements.ohlcv.add(self._symbol, "1d")


class StrategyFixture(CSVOHLCVStrategyFixture, StrategyProtocol):
    market_type = "futures"

    def __init__(self, csv: io.StringIO, symbol: Symbol, handler: Callable):
        super().__init__(csv, symbol)
        self.handler = handler
        self.count = 0

    def book_trading_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: set[TimeFrame],
    ) -> None:
        self.handler(self.count, timestamp, bookkeeper, triggered_timeframes)
        self.count += 1


@pytest.fixture
def make_strategy():
    def factory(csv: io.StringIO, symbol: Symbol, handler: Callable) -> StrategyFixture:
        return StrategyFixture(csv, symbol, handler)

    return factory


@pytest.fixture
def run_backtest(account):
    def runner(
        strategy: StrategyFixture,
        simulation_engine: object,
        fill_recorder: CacheFillRecorder,
    ) -> None:
        backtester = Backtester(
            strategy,
            simulation_engine,  # type: ignore[arg-type]
            strategy.ohlcv_provider,
            fill_recorder,
            "2024-01-01",
            "2024-01-03",
        )
        asyncio.run(backtester.run())

    return runner
