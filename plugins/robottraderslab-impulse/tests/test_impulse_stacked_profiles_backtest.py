import io
from collections.abc import Callable
from typing import Any

import pytest
from impulse_helpers import new_exchange, run_scripted_backtest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
)
from robottraderslab.strategies import PositionSide

BTC = Symbol.create("BTC/USDT:USDT")
TIMEFRAME = "1d"
END_DATE = "2024-01-05"
ALPHA = "alpha"
BETA = "beta"
ALPHA_PROFILE_ID = f"{TIMEFRAME}-{ALPHA}"
BETA_PROFILE_ID = f"{TIMEFRAME}-{BETA}"

type ScriptedSignals = dict[str, dict[int, str]]
type RunBacktest = Callable[[list[dict[str, Any]], ScriptedSignals], None]


@pytest.fixture
def simulated_exchange() -> SimulatedFuturesExchange:
    return new_exchange(SimulatedFuturesExchange)


@pytest.fixture
def fill_recorder(simulated_exchange: SimulatedFuturesExchange) -> CacheFillRecorder:
    return simulated_exchange.fill_recorder


@pytest.fixture
def simulation_engine(
    simulated_exchange: SimulatedFuturesExchange,
) -> FuturesSimulationEngine:
    return simulated_exchange.simulation_engine


@pytest.fixture
def flat_prices() -> io.StringIO:
    """Five candles at a steady price, so sizes stay a plain share of the balance."""
    return io.StringIO("""\
date,open,high,low,close,volume
2024-01-01T00:00:00+00:00,100,101,99,100,10
2024-01-02T00:00:00+00:00,100,101,99,100,10
2024-01-03T00:00:00+00:00,100,101,99,100,10
2024-01-04T00:00:00+00:00,100,101,99,100,10
2024-01-05T00:00:00+00:00,100,101,99,100,10
""")


@pytest.fixture
def run_backtest(
    simulated_exchange: SimulatedFuturesExchange, flat_prices: io.StringIO
) -> RunBacktest:
    """Return a callable running one backtest over the scripted signals."""

    def _run(profiles: list[dict[str, Any]], signals_by_tag: ScriptedSignals) -> None:
        run_scripted_backtest(
            simulated_exchange, signals_by_tag, profiles, flat_prices, END_DATE
        )

    return _run


class TestStackedProfiles:
    """Two profiles on one symbol, held over several candles.

    The venue is one-way, so it nets every order into the single position it
    holds. At the end of any candle that net has to equal the sum of what the
    profiles believe they hold, whether a profile contributes a fresh order or
    a position it opened candles ago.
    """

    def test_opposite_entries_on_the_same_candle(self, run_backtest, simulation_engine):
        run_backtest(
            _stacked_profiles(alpha_ratio=0.5, beta_ratio=0.25),
            {ALPHA: {1: "long_entry"}, BETA: {1: "short_entry"}},
        )

        held = simulation_engine.open_positions[BTC]
        assert held.side == PositionSide.LONG
        assert held.quantity == pytest.approx(25.0)

    def test_one_profile_exits_while_the_other_still_holds(
        self, run_backtest, simulation_engine
    ):
        run_backtest(
            _stacked_profiles(alpha_ratio=0.5, beta_ratio=0.25),
            {ALPHA: {1: "long_entry", 3: "long_exit"}, BETA: {1: "short_entry"}},
        )

        held = simulation_engine.open_positions[BTC]
        assert held.side == PositionSide.SHORT
        assert held.quantity == pytest.approx(25.0)

    def test_an_exit_after_offsetting_profiles_left_the_venue_flat(
        self, run_backtest, fill_recorder
    ):
        run_backtest(
            _stacked_profiles(alpha_ratio=0.25, beta_ratio=0.25),
            {
                ALPHA: {1: "long_entry", 3: "long_exit"},
                BETA: {1: "short_entry", 4: "short_exit"},
            },
        )

        assert _recorded_fills(fill_recorder) == [
            (ALPHA_PROFILE_ID, "enter_long", 25.0),
            (BETA_PROFILE_ID, "exit_long", 25.0),
            (ALPHA_PROFILE_ID, "enter_short", 25.0),
            (BETA_PROFILE_ID, "exit_short", 25.0),
        ]


def _stacked_profiles(*, alpha_ratio: float, beta_ratio: float) -> list[dict[str, Any]]:
    return [
        {
            "symbol": str(BTC),
            "timeframe": TIMEFRAME,
            "trix_length": 2,
            "signal_length": 2,
            "trend_length": 2,
            "leverage": 1.0,
            "tag": tag,
            "total_balance_ratio": total_balance_ratio,
        }
        for tag, total_balance_ratio in ((ALPHA, alpha_ratio), (BETA, beta_ratio))
    ]


def _recorded_fills(
    fill_recorder: CacheFillRecorder,
) -> list[tuple[str, str, float]]:
    """Return each fill as (setup name, fill type, quantity), in order."""
    recorded = fill_recorder.get_fills()
    return [
        (str(tag), str(fill_type), float(quantity))
        for tag, fill_type, quantity in zip(
            recorded["tag"],
            recorded["fill_type"],
            recorded["net_quantity"],
            strict=True,
        )
    ]
