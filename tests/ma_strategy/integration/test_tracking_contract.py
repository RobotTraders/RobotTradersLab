import asyncio
import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab._core import (
    AccountRequirements,
    FeeMode,
    PlacementReserve,
    exchange_position_tracker,
    fetch_account_snapshots,
)
from robottraderslab.backtester.backtester import Backtester
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FeeRates,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
    fee_model_for,
)
from robottraderslab.ma_strategy.futures_ma_strategy import (
    FuturesMAStrategy,
    ProfileConfig,
)
from robottraderslab.ohlcv_provider import CSVOHLCVProvider
from robottraderslab.strategies import (
    PositionTracker,
    StrategyRequirements,
    TrackedPosition,
    TrackingId,
    TradingSystem,
)
from robottraderslab.strategies.futures import FuturesAccount

BTC = Symbol.create("BTC/USDT:USDT")
TIMEFRAME: TimeFrame = "1d"

_INTEGRATION_DATA = Path(__file__).parent
_RESIZING_PRICES = _INTEGRATION_DATA / "multi_profile" / "data" / "resizing_test.csv"
_BTC_2024_PRICES = (
    _INTEGRATION_DATA / "single_symbol_pipeline" / "data" / "btc_usdt_binance_2024.csv"
)

_INITIAL_BALANCE = 10_000.0
_FEE_RATE = 0.001
_MARGIN_CURRENCY = "USDT"
_NO_RESERVE = PlacementReserve(margin_markup=0.0, fee_markup=0.0, notional_reserve=0.0)
_TRACKER_LOGGER = exchange_position_tracker.__name__

_ALPHA_LONG_QUANTITY = {"cost": 24.01442308, "exchange": 24.01444709}
_BETA_LONG_QUANTITY = {"cost": 16.43092105, "exchange": 16.43093748}
_QUANTITY_AFTER_RE_ENTRY = {"cost": 21.66323708, "exchange": 21.66326338}
_BTC_2024_FILLS = 48
_QUANTITY_TOLERANCE = 1e-12

_CROSSOVER_UP_AND_DOWN_AND_UP = """\
date,open,high,low,close,volume
2024-01-01 00:00:00+0000,100,101,99,100,1000
2024-01-02 00:00:00+0000,100,101,89,90,1000
2024-01-03 00:00:00+0000,90,91,79,80,1000
2024-01-04 00:00:00+0000,80,112,79,110,1000
2024-01-05 00:00:00+0000,110,116,108,115,1000
2024-01-06 00:00:00+0000,115,116,89,90,1000
2024-01-07 00:00:00+0000,90,91,69,70,1000
2024-01-08 00:00:00+0000,70,101,69,100,1000
2024-01-09 00:00:00+0000,100,111,99,110,1000
2024-01-10 00:00:00+0000,110,116,108,115,1000
"""

_CROSSOVER_UP_THEN_A_WICK_THROUGH_LIQUIDATION = """\
date,open,high,low,close,volume
2024-01-01 00:00:00+0000,100,101,99,100,1000
2024-01-02 00:00:00+0000,100,101,89,90,1000
2024-01-03 00:00:00+0000,90,91,79,80,1000
2024-01-04 00:00:00+0000,80,112,105,110,1000
2024-01-05 00:00:00+0000,110,116,108,115,1000
2024-01-06 00:00:00+0000,115,121,40,120,1000
2024-01-07 00:00:00+0000,120,121,118,120,1000
2024-01-08 00:00:00+0000,120,121,118,120,1000
"""

type PriceSource = Path | StringIO
type ProfileFactory = Callable[..., dict[str, Any]]
type RunScenario = Callable[..., "ContractRun"]


class ContractRun:
    """What one backtest left behind, read back the way a scenario asserts on it."""

    def __init__(
        self,
        tracker: PositionTracker,
        profiles: Sequence[ProfileConfig],
        simulation_engine: FuturesSimulationEngine,
        fills: Any,
        reconciliations: list[str],
    ) -> None:
        self._tracker = tracker
        self._profiles = profiles
        self.venue = simulation_engine.open_positions
        self.fills = fills
        self.reconciliations = reconciliations

    @property
    def fill_types(self) -> list[str]:
        return list(self.fills["fill_type"])

    @property
    def profile_ids(self) -> dict[str, TrackingId]:
        return {profile.tag: profile.profile_id for profile in self._profiles}

    @property
    def tracked(self) -> dict[str, TrackedPosition]:
        return {
            profile.tag: held
            for profile in self._profiles
            if (held := self._tracker.get(profile.profile_id)) is not None
        }


_ALL_FEE_MODES: list[FeeMode] = ["cost", "exchange"]


@pytest.fixture(params=_ALL_FEE_MODES)
def fee_mode(request: pytest.FixtureRequest) -> FeeMode:
    """The conventions a backtest can charge its fees under; the tracking
    contract holds under either, since a profile tracks what was filled.
    """
    return request.param


@pytest.fixture
def ma_profile() -> ProfileFactory:
    """A profile's tag is what separates its fills and its tracked position
    from those of the profiles it shares a symbol with.

    Passing a `trend_ma_length` trades both directions, filtered by the trend
    it sets; leaving it unset trades the crossover alone.
    """

    def _build(
        tag: str,
        *,
        fast_ma_length: int,
        slow_ma_length: int,
        available_balance_ratio: float,
        trend_ma_length: int | None = None,
        long_only: bool = False,
        leverage: float = 1.0,
    ) -> dict[str, Any]:
        return {
            "symbol": str(BTC),
            "timeframe": TIMEFRAME,
            "tag": tag,
            "fast_ma_length": fast_ma_length,
            "slow_ma_length": slow_ma_length,
            "trend_ma_length": trend_ma_length,
            "available_balance_ratio": available_balance_ratio,
            "long_only": long_only,
            "leverage": leverage,
        }

    return _build


@pytest.fixture
def resizing_profiles(ma_profile: ProfileFactory) -> list[dict[str, Any]]:
    return [
        ma_profile(
            "alpha",
            fast_ma_length=2,
            slow_ma_length=3,
            trend_ma_length=6,
            available_balance_ratio=0.25,
        ),
        ma_profile(
            "beta",
            fast_ma_length=3,
            slow_ma_length=4,
            trend_ma_length=6,
            available_balance_ratio=0.25,
        ),
    ]


@pytest.fixture
def run_scenario(
    fee_mode: FeeMode,
    caplog: pytest.LogCaptureFixture,
    backtest_trading_system: TradingSystem,
) -> RunScenario:
    """Fees are charged, so a profile tracks the quantity the simulator
    reported filled, never the one its order asked for.
    """

    def _run(
        profiles: Sequence[dict[str, Any]],
        *,
        prices: PriceSource,
        start_date: str,
        end_date: str,
    ) -> ContractRun:
        fill_recorder = CacheFillRecorder()
        simulation_engine = FuturesSimulationEngine(
            initial_balance={_MARGIN_CURRENCY: _INITIAL_BALANCE},
            fee_rates=FeeRates(maker=_FEE_RATE, taker=_FEE_RATE),
            fill_recorder=fill_recorder,
            margin_currency=_MARGIN_CURRENCY,
            fee_model=fee_model_for(fee_mode, _NO_RESERVE),
        )
        exchange = SimulatedFuturesExchange(simulation_engine, fill_recorder)
        strategy = FuturesMAStrategy(
            account=FuturesAccount(exchange),
            trading_system=backtest_trading_system,
            config_dir=Path("bot-config-dir"),
            profiles=list(profiles),
        )
        backtester = Backtester(
            strategy,
            simulation_engine,
            CSVOHLCVProvider(file=prices, symbol=BTC, timeframe=TIMEFRAME),
            fill_recorder,
            start_date,
            end_date,
            requirements_factory=StrategyRequirements,
        )

        with caplog.at_level(logging.WARNING, logger=_TRACKER_LOGGER):
            outputs = asyncio.run(backtester.run())

        _refresh_as_the_next_cycle_would(strategy, start_date)

        return ContractRun(
            tracker=strategy._position_tracker,
            profiles=list(strategy.profiles),
            simulation_engine=simulation_engine,
            fills=outputs.fills,
            reconciliations=[
                record.getMessage()
                for record in caplog.records
                if record.name == _TRACKER_LOGGER
            ],
        )

    return _run


class TestProfileFillsSumToTheVenue:
    """A profile holds what its own fills reported filled, and no more.

    The venue nets every profile on a symbol into one position, so the
    profiles are honest exactly while their signed sum reproduces that
    position with no correction from the engine.
    """

    def test_one_profile_round_trips(self, run_scenario, resizing_profiles):
        run = run_scenario(
            resizing_profiles[:1],
            prices=_RESIZING_PRICES,
            start_date="2024-01-01",
            end_date="2024-01-30",
        )

        assert run.reconciliations == []
        assert run.fill_types == [
            "enter_long",
            "exit_long",
            "enter_short",
            "exit_short",
        ]
        assert run.tracked == {}
        assert run.venue == {}

    def test_two_profiles_stack_on_the_same_side(
        self, run_scenario, resizing_profiles, fee_mode
    ):
        run = run_scenario(
            resizing_profiles,
            prices=_RESIZING_PRICES,
            start_date="2024-01-01",
            end_date="2024-01-12",
        )

        assert run.reconciliations == []
        assert run.fill_types == ["enter_long", "add_to_long"]
        assert run.tracked["alpha"].quantity == pytest.approx(
            _ALPHA_LONG_QUANTITY[fee_mode], rel=_QUANTITY_TOLERANCE
        )
        assert run.tracked["beta"].quantity == pytest.approx(
            _BETA_LONG_QUANTITY[fee_mode], rel=_QUANTITY_TOLERANCE
        )
        assert run.venue[BTC].quantity == pytest.approx(
            _ALPHA_LONG_QUANTITY[fee_mode] + _BETA_LONG_QUANTITY[fee_mode],
            rel=_QUANTITY_TOLERANCE,
        )

    def test_one_profile_exits_while_the_other_still_holds(
        self, run_scenario, resizing_profiles, fee_mode
    ):
        run = run_scenario(
            resizing_profiles,
            prices=_RESIZING_PRICES,
            start_date="2024-01-01",
            end_date="2024-01-14",
        )

        assert run.reconciliations == []
        assert run.fill_types == ["enter_long", "add_to_long", "reduce_long"]
        assert set(run.tracked) == {"beta"}
        assert run.tracked["beta"].quantity == pytest.approx(
            _BETA_LONG_QUANTITY[fee_mode], rel=_QUANTITY_TOLERANCE
        )
        assert run.venue[BTC].quantity == pytest.approx(
            _BETA_LONG_QUANTITY[fee_mode], rel=_QUANTITY_TOLERANCE
        )

    def test_one_profile_re_enters_after_closing(
        self, run_scenario, ma_profile, fee_mode
    ):
        run = run_scenario(
            [
                ma_profile(
                    "alpha",
                    fast_ma_length=2,
                    slow_ma_length=3,
                    available_balance_ratio=0.25,
                    long_only=True,
                )
            ],
            prices=StringIO(_CROSSOVER_UP_AND_DOWN_AND_UP),
            start_date="2024-01-01",
            end_date="2024-01-10",
        )

        assert run.reconciliations == []
        assert run.fill_types == ["enter_long", "exit_long", "enter_long"]
        assert run.tracked["alpha"].quantity == pytest.approx(
            _QUANTITY_AFTER_RE_ENTRY[fee_mode], rel=_QUANTITY_TOLERANCE
        )
        assert run.venue[BTC].quantity == pytest.approx(
            _QUANTITY_AFTER_RE_ENTRY[fee_mode], rel=_QUANTITY_TOLERANCE
        )

    def test_a_year_of_stacked_trading_needs_no_correction(
        self, run_scenario, ma_profile
    ):
        run = run_scenario(
            [
                ma_profile(
                    "fast",
                    fast_ma_length=3,
                    slow_ma_length=10,
                    trend_ma_length=40,
                    available_balance_ratio=0.25,
                ),
                ma_profile(
                    "slow",
                    fast_ma_length=7,
                    slow_ma_length=15,
                    trend_ma_length=40,
                    available_balance_ratio=0.25,
                ),
            ],
            prices=_BTC_2024_PRICES,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert run.reconciliations == []
        assert len(run.fills) == _BTC_2024_FILLS
        assert run.tracked == {}
        assert run.venue == {}


class TestVenueFiredEvents:
    """An event no profile fill produced leaves a symbol nobody can claim.

    The venue closes the whole netted position at once, so every profile
    holding that symbol loses its position together, whatever share of it each
    had opened. The tracker folds the venue's own closing fill and has nothing to
    correct.
    """

    def test_a_liquidation_clears_a_symbol_two_profiles_hold(
        self, run_scenario, ma_profile
    ):
        run = run_scenario(
            [
                ma_profile(
                    "alpha",
                    fast_ma_length=2,
                    slow_ma_length=3,
                    available_balance_ratio=0.3,
                    long_only=True,
                    leverage=10.0,
                ),
                ma_profile(
                    "beta",
                    fast_ma_length=2,
                    slow_ma_length=3,
                    available_balance_ratio=0.2,
                    long_only=True,
                    leverage=10.0,
                ),
            ],
            prices=StringIO(_CROSSOVER_UP_THEN_A_WICK_THROUGH_LIQUIDATION),
            start_date="2024-01-01",
            end_date="2024-01-08",
        )

        assert run.fill_types == ["enter_long", "add_to_long", "liquidate_long"]
        assert run.reconciliations == []
        assert run.tracked == {}
        assert run.venue == {}


def _refresh_as_the_next_cycle_would(
    strategy: FuturesMAStrategy, start_date: str
) -> None:
    """A fill lands after its candle's refresh has already run, so the run's
    last fills belong to the refresh of a cycle the backtest never reaches.
    The record is read the way live reads it: settled by that next refresh.
    """
    venue_reads = AccountRequirements()
    venue_reads.add(strategy.account, symbols=(BTC,), positions=True, executions=True)
    run_start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    snapshots = asyncio.run(
        fetch_account_snapshots(venue_reads, executions_since=run_start)
    )
    strategy._position_tracker.refresh(
        snapshots.of(strategy.account), run_start, datetime.now(timezone.utc)
    )
