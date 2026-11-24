import asyncio
import io
from datetime import datetime

from robottraderslab import Symbol, TimeFrame
from robottraderslab.backtester.backtester import Backtester
from robottraderslab.backtester.simulator import SimulatedFuturesExchange
from robottraderslab.exchanges import MarginMode, MarginSettings
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

TWO_DAYS = """\
date,open,high,low,close,volume
2024-01-01T00:00:00+00:00,100,110,90,105,10
2024-01-02T00:00:00+00:00,105,115,95,110,10
"""


class _LeverageReadingStrategy(StrategyProtocol):
    market_type = "futures"

    def __init__(self, exchange: SimulatedFuturesExchange):
        self._exchange = exchange
        self.account = FuturesAccount(exchange)
        self.ohlcv_provider = CSVOHLCVProvider(
            file=io.StringIO(TWO_DAYS), symbol=str(BTCUSDT_PERP), timeframe="1d"
        )
        self.simulated_leverages: list[float] = []
        self.snapshot_leverages: list[float | None] = []

    async def setup(self, requirements: StrategyRequirements) -> None:
        requirements.ohlcv.add(BTCUSDT_PERP, "1d")
        requirements.account.add(
            self.account,
            margin_targets={
                BTCUSDT_PERP: MarginSettings(
                    leverage=5.0, margin_mode=MarginMode.ISOLATED
                )
            },
        )

    def generate_trading_signals(self, ohlcvs: OHLCVs) -> None: ...

    def book_trading_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: set[TimeFrame],
    ) -> None:
        engine = self._exchange.simulation_engine
        self.simulated_leverages.append(engine.get_symbol_leverage(BTCUSDT_PERP))
        settings = account_snapshots.of(self.account).margin_settings(BTCUSDT_PERP)
        self.snapshot_leverages.append(settings.leverage)


def test_a_backtest_starts_at_the_declared_leverage():
    exchange = SimulatedFuturesExchange.create_from_settings(
        initial_balance={"USDT": 10_000.0}, maker_fee_rate=0.0, taker_fee_rate=0.0
    )
    strategy = _LeverageReadingStrategy(exchange)
    backtester = Backtester(
        strategy,
        exchange.simulation_engine,
        strategy.ohlcv_provider,
        exchange.fill_recorder,
        "2024-01-01",
        "2024-01-03",
    )

    asyncio.run(backtester.run())

    assert strategy.simulated_leverages[0] == 5.0
    assert strategy.snapshot_leverages[0] == 5.0
