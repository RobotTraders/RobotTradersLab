import io
from datetime import datetime, timezone

import pytest
from impulse_helpers import new_exchange, run_scripted_backtest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
)

BTC = Symbol.create("BTC/USDT:USDT")
TIMEFRAME = "1d"
END_DATE = "2024-01-03"


@pytest.fixture
def simulated_exchange() -> SimulatedFuturesExchange:
    exchange = new_exchange(SimulatedFuturesExchange)
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
def crash_after_entry_prices() -> io.StringIO:
    return io.StringIO("""\
date,open,high,low,close,volume
2024-01-01T00:00:00+00:00,100,101,99,100,10
2024-01-02T00:00:00+00:00,100,101,99,100,10
2024-01-03T00:00:00+00:00,95,95,80,85,10
""")


class TestStopLossBacktest:
    def test_stop_loss_fires_and_closes_net_position(
        self,
        simulated_exchange,
        simulation_engine,
        fill_recorder,
        crash_after_entry_prices,
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
        run_scripted_backtest(
            simulated_exchange,
            {"": {1: "long_entry"}},
            [profile],
            crash_after_entry_prices,
            END_DATE,
        )

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
