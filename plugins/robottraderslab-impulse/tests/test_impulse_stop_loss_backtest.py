import asyncio
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from impulse_helpers import write_signal_columns

from robottraderslab import Symbol
from robottraderslab.backtester.backtester import Backtester
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
)
from robottraderslab.ohlcv_provider import CSVOHLCVProvider
from robottraderslab.strategies import OHLCVs, TradingMode, TradingSystem
from robottraderslab.strategies.futures import FuturesAccount
from robottraderslab_impulse import ImpulseStrategy
from robottraderslab_impulse.profile_config import ProfileConfig

BTC = Symbol.create("BTC/USDT:USDT")
TIMEFRAME = "1d"
START_DATE = "2024-01-01"
END_DATE = "2024-01-03"


class _ScriptedImpulse(ImpulseStrategy):
    """Impulse strategy with long entries injected on fixed bars, bypassing TRIX."""

    def __init__(self, *, entry_bars: set[int], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._entry_bars = entry_bars

    def generate_profile_signals(self, profile: ProfileConfig, ohlcvs: OHLCVs) -> None:
        write_signal_columns(
            ohlcvs,
            profile.symbol,
            profile.timeframe,
            profile.profile_id,
            dict.fromkeys(self._entry_bars, "long_entry"),
        )


@pytest.fixture
def simulated_exchange() -> SimulatedFuturesExchange:
    exchange = SimulatedFuturesExchange.create_from_settings(
        initial_balance={"USDT": 10_000.0},
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        fee_mode="cost",
    )
    exchange.simulation_engine.enable_execution_recording()
    return exchange


@pytest.fixture
def fill_recorder(simulated_exchange: SimulatedFuturesExchange) -> CacheFillRecorder:
    return simulated_exchange.fill_recorder


@pytest.fixture
def simulation_engine(
    simulated_exchange: SimulatedFuturesExchange,
) -> FuturesSimulationEngine:
    return simulated_exchange.simulation_engine


@pytest.fixture
def account(simulated_exchange: SimulatedFuturesExchange) -> FuturesAccount:
    return FuturesAccount(simulated_exchange)


@pytest.fixture
def crash_after_entry_prices() -> io.StringIO:
    return io.StringIO("""\
date,open,high,low,close,volume
2024-01-01T00:00:00+00:00,100,101,99,100,10
2024-01-02T00:00:00+00:00,100,101,99,100,10
2024-01-03T00:00:00+00:00,95,95,80,85,10
""")


class TestStopLossBacktest:
    def test_stop_loss_fires_and_closes_net_position(
        self, account, simulation_engine, fill_recorder, crash_after_entry_prices
    ):
        """A long entry's stop-loss fires when price crashes through the trigger."""
        profile = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": TIMEFRAME,
            "trix_length": 2,
            "signal_length": 2,
            "trend_length": 2,
            "total_balance_ratio": 0.5,
            "leverage": 1.0,
            "stop_loss_pct": 0.1,
        }
        strategy = _ScriptedImpulse(
            account=account,
            trading_system=TradingSystem(trading_mode=TradingMode.BACKTEST),
            config_dir=Path("bot-config-dir"),
            profiles=[profile],
            entry_bars={1},
        )
        provider = CSVOHLCVProvider(
            file=crash_after_entry_prices,
            symbol=str(BTC),
            timeframe=TIMEFRAME,
        )
        backtester = Backtester(
            strategy,
            simulation_engine,
            provider,
            fill_recorder,
            START_DATE,
            END_DATE,
        )

        asyncio.run(backtester.run())

        epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
        executions = simulation_engine.get_executions_since(epoch)
        stop_loss_executions = [
            execution for execution in executions if execution.kind == "stop-loss"
        ]
        fills = fill_recorder.get_fills()
        assert len(stop_loss_executions) == 1
        assert stop_loss_executions[0].symbol == BTC
        assert simulation_engine.open_positions == {}
        assert fills.iloc[-1]["price"] == pytest.approx(90.0)
