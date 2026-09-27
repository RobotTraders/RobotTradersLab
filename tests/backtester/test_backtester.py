import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest import mock
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab._core import (
    OHLCVProviderProtocol,
    OHLCVRequirement,
    PlacementReserve,
    TimeframeSnapshot,
)
from robottraderslab.backtester.backtester import Backtester, _executions_window_seconds
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FeeRates,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
    fee_model_for,
)
from robottraderslab.exceptions import ExchangeCriticalError, StrategyCriticalError
from robottraderslab.exchanges import OrderFill, OrderSide, PlacedOrder
from robottraderslab.futures import FuturesAccount
from robottraderslab.ohlcv_provider import MockOHLCVProvider
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    OHLCVs,
    ProfileStrategy,
    StrategyProtocol,
    TrackingId,
    TradingMode,
    TradingSystem,
    profile_tag,
)


@dataclass(frozen=True, slots=True)
class _Profile:
    profile_id: str
    timeframe: TimeFrame


@pytest.fixture
def fill_recorder() -> CacheFillRecorder:
    return CacheFillRecorder()


@pytest.fixture
def simulation_engine(fill_recorder: CacheFillRecorder) -> FuturesSimulationEngine:
    margin_currency = "USDT"
    initial_balance = {margin_currency: 100_000.0}
    fee_rates = FeeRates(maker=0.001, taker=0.002)
    return FuturesSimulationEngine(
        initial_balance,
        fee_rates,
        fill_recorder=fill_recorder,
        margin_currency=margin_currency,
        fee_model=fee_model_for("cost", PlacementReserve()),
    )


@pytest.fixture
def account(
    simulation_engine: FuturesSimulationEngine, fill_recorder: CacheFillRecorder
) -> FuturesAccount:
    return FuturesAccount(SimulatedFuturesExchange(simulation_engine, fill_recorder))


@pytest.fixture
def create_timeframe_snapshot():
    def _create_timeframe_snapshot(symbol: Symbol):
        timestamp = datetime.now()
        ohlcv_df = pd.DataFrame(
            {
                "open": [63309.2],
                "high": [64097.6],
                "low": [60136.7],
                "close": [60790.0],
                "volume": [307622.802],
            },
            index=[timestamp],
        )

        timeframe_df = pd.concat(
            [ohlcv_df],
            axis=1,
            keys=[str(symbol)],
            names=["symbol", "metric"],
        )

        return TimeframeSnapshot(
            timestamp,
            ["1d"],
            {"1d": timeframe_df},
            {"1d": pd.DatetimeIndex([timestamp])},
            {"1d": [0]},
            0,
            {},
        )

    return _create_timeframe_snapshot


@pytest.fixture
def create_strategy_place_market_order(
    account: FuturesAccount, create_timeframe_snapshot
):
    def _create_strategy(symbol: Symbol, on_filled) -> StrategyProtocol:
        class Strategy(StrategyProtocol):
            market_type = "futures"

            def __init__(self):
                self._snapshot = create_timeframe_snapshot(symbol)

            async def setup(self, requirements):
                requirements.ohlcv.add(symbol, "1d")

            def generate_trading_signals(self, ohlcvs: OHLCVs) -> None:
                pass

            def book_trading_actions(
                self,
                ohlcvs: OHLCVs,
                _snapshots: AccountSnapshots,
                _timestamp: datetime,
                bookkeeper: BookKeeper,
                _triggered_timeframes: set[TimeFrame],
            ) -> None:
                action = account.long_entry(symbol, 1.0).when_filled(on_filled).build()
                bookkeeper.add(action)

        return Strategy()

    return _create_strategy


class MyCustomException(Exception): ...


def test_calls_setup_before_generate_trading_signals(
    simulation_engine, fill_recorder, account
):
    symbol = Symbol.create("BTC/USDT:USDT")

    async def add_requirements(requirements):
        requirements.ohlcv.add(symbol, "1d")

    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)
    strategy.generate_trading_signals.side_effect = MyCustomException()

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1000],
        },
        index=[pd.Timestamp("2024-01-01", tz="UTC")],
    )
    backtester = Backtester(
        strategy,
        simulation_engine,
        ohlcv_provider,
        fill_recorder,
        "2024-01-01",
        "2024-01-01",
    )

    try:
        asyncio.run(backtester.run())
        raise AssertionError("Expect MyCustomException to be raised")
    except MyCustomException:
        strategy.setup.assert_called_once()


def test_logs_the_profile_strategy_once_over_every_candle(
    simulation_engine, fill_recorder, caplog
):
    symbol = Symbol.create("BTC/USDT:USDT")

    class Strategy(ProfileStrategy[_Profile]):
        market_type = "futures"

        def __init__(self):
            super().__init__(
                account=object(),
                trading_system=TradingSystem(trading_mode=TradingMode.BACKTEST),
                config_dir=None,
                profiles=[{"profile_id": "alpha", "timeframe": "1d"}],
            )
            self.booked: list[datetime] = []

        async def setup(self, requirements):
            requirements.ohlcv.add(symbol, "1d")

        def book_profile_actions(
            self, profile, ohlcvs, account_snapshots, timestamp, bookkeeper
        ):
            self.booked.append(timestamp)

    strategy = Strategy()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "open": [100.0] * 4,
            "high": [110.0] * 4,
            "low": [90.0] * 4,
            "close": [105.0] * 4,
            "volume": [1000] * 4,
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
    )
    backtester = Backtester(
        strategy,
        simulation_engine,
        ohlcv_provider,
        fill_recorder,
        "2024-01-01",
        "2024-01-04",
    )

    with caplog.at_level(logging.INFO):
        asyncio.run(backtester.run())

    assert len(strategy.booked) == 4
    assert caplog.messages.count("Strategy: 1 profile; slots: per-profile booking") == 1


def test_calls_on_filled_callbacks(
    simulation_engine, fill_recorder, account, create_strategy_place_market_order
):
    symbol = Symbol.create("BTC/USDT:USDT")
    on_filled = AsyncMock()
    strategy = create_strategy_place_market_order(symbol, on_filled)

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "open": [60790.0],
            "high": [64097.6],
            "low": [60136.7],
            "close": [60790.0],
            "volume": [307622.802],
        },
        index=[pd.Timestamp("2024-01-01", tz="UTC")],
    )
    backtester = Backtester(
        strategy,
        simulation_engine,
        ohlcv_provider,
        fill_recorder,
        "2024-01-01",
        "2024-01-01",
    )

    asyncio.run(backtester.run())

    on_filled.assert_awaited_once_with(
        PlacedOrder(order_id=mock.ANY, get_fill=mock.ANY),
        OrderFill(
            order_id=mock.ANY,
            symbol=symbol,
            side=OrderSide.BUY,
            kind="market",
            quantity=0.998,
            timestamp=mock.ANY,
            source="strategy",
        ),
    )


@pytest.mark.parametrize(
    "error",
    [
        ExchangeCriticalError("Auth expired"),
        StrategyCriticalError("Auth expired"),
    ],
    ids=["exchange_critical", "strategy_critical"],
)
def test_with_critical_error_in_callback(
    simulation_engine, fill_recorder, account, create_strategy_place_market_order, error
):
    symbol = Symbol.create("BTC/USDT:USDT")
    on_all_actions_executed = AsyncMock(side_effect=error)
    strategy = create_strategy_place_market_order(symbol, on_all_actions_executed)

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "open": [60790.0],
            "high": [64097.6],
            "low": [60136.7],
            "close": [60790.0],
            "volume": [307622.802],
        },
        index=[pd.Timestamp("2024-01-01", tz="UTC")],
    )
    backtester = Backtester(
        strategy,
        simulation_engine,
        ohlcv_provider,
        fill_recorder,
        "2024-01-01",
        "2024-01-01",
    )

    with pytest.raises(type(error), match="Auth expired"):
        asyncio.run(backtester.run())


def test_run_with_no_requirements_raises(simulation_engine, fill_recorder, account):
    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock()
    ohlcv_provider = Mock(spec=MockOHLCVProvider)

    backtester = Backtester(
        strategy,
        simulation_engine,
        ohlcv_provider,
        fill_recorder,
        "2024-01-01",
        "2024-01-01",
    )

    with pytest.raises(StrategyCriticalError, match="declared no OHLCV requirements"):
        asyncio.run(backtester.run())


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Transient network error"),
        asyncio.CancelledError("Callback cancelled itself"),
    ],
    ids=["generic_error", "base_exception"],
)
def test_non_critical_callback_exception(
    simulation_engine,
    fill_recorder,
    account,
    create_strategy_place_market_order,
    caplog,
    error,
):
    symbol = Symbol.create("BTC/USDT:USDT")
    on_filled = AsyncMock(side_effect=error)
    strategy = create_strategy_place_market_order(symbol, on_filled)

    ohlcv_provider = Mock(spec=MockOHLCVProvider)
    ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "open": [60790.0],
            "high": [64097.6],
            "low": [60136.7],
            "close": [60790.0],
            "volume": [307622.802],
        },
        index=[pd.Timestamp("2024-01-01", tz="UTC")],
    )
    backtester = Backtester(
        strategy,
        simulation_engine,
        ohlcv_provider,
        fill_recorder,
        "2024-01-01",
        "2024-01-01",
    )

    with caplog.at_level(logging.ERROR):
        asyncio.run(backtester.run())

    on_filled.assert_awaited_once()
    assert "Failed to invoke callback" in caplog.text


class TestExecutionRecording:
    def test_declaring_no_executions_leaves_recording_off(
        self,
        simulation_engine,
        fill_recorder,
        account,
        create_strategy_place_market_order,
    ):
        symbol = Symbol.create("BTC/USDT:USDT")
        strategy = create_strategy_place_market_order(symbol, AsyncMock())
        ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
        ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
            {
                "open": [60790.0],
                "high": [64097.6],
                "low": [60136.7],
                "close": [60790.0],
                "volume": [307622.802],
            },
            index=[pd.Timestamp("2024-01-01", tz="UTC")],
        )
        backtester = Backtester(
            strategy,
            simulation_engine,
            ohlcv_provider,
            fill_recorder,
            "2024-01-01",
            "2024-01-01",
        )

        asyncio.run(backtester.run())

        assert (
            simulation_engine.get_executions_since(
                datetime.min.replace(tzinfo=timezone.utc)
            )
            == []
        )

    def test_declaring_executions_switches_recording_on(
        self, simulation_engine, fill_recorder, account
    ):
        symbol = Symbol.create("BTC/USDT:USDT")

        class Strategy(StrategyProtocol):
            market_type = "futures"

            async def setup(self, requirements):
                requirements.ohlcv.add(symbol, "1d")
                requirements.account.add(account, symbols=[symbol], executions=True)

            def generate_trading_signals(self, ohlcvs: OHLCVs) -> None:
                pass

            def book_trading_actions(
                self,
                ohlcvs: OHLCVs,
                _snapshots: AccountSnapshots,
                _timestamp: datetime,
                bookkeeper: BookKeeper,
                _triggered_timeframes: set[TimeFrame],
            ) -> None:
                bookkeeper.add(account.long_entry(symbol, 1.0).build())

        ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
        ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
            {
                "open": [60790.0],
                "high": [64097.6],
                "low": [60136.7],
                "close": [60790.0],
                "volume": [307622.802],
            },
            index=[pd.Timestamp("2024-01-01", tz="UTC")],
        )
        backtester = Backtester(
            Strategy(),
            simulation_engine,
            ohlcv_provider,
            fill_recorder,
            "2024-01-01",
            "2024-01-01",
        )

        asyncio.run(backtester.run())

        assert (
            simulation_engine.get_executions_since(
                datetime.min.replace(tzinfo=timezone.utc)
            )
            != []
        )


class TestExecutionsWindowSlidesByLookback:
    def test_a_later_candle_stops_seeing_an_execution_outside_its_lookback(
        self, simulation_engine, fill_recorder, account
    ):
        symbol = Symbol.create("BTC/USDT:USDT")
        lookback = 3
        seen_by_candle_index: dict[int, list] = {}

        class Strategy(StrategyProtocol):
            market_type = "futures"

            def __init__(self) -> None:
                self._candle_index = 0

            async def setup(self, requirements) -> None:
                requirements.ohlcv.add(symbol, "1d", lookback)
                requirements.account.add(account, symbols=[symbol], executions=True)

            def generate_trading_signals(self, ohlcvs: OHLCVs) -> None:
                pass

            def book_trading_actions(
                self,
                _ohlcvs: OHLCVs,
                snapshots: AccountSnapshots,
                _timestamp: datetime,
                bookkeeper: BookKeeper,
                _triggered_timeframes: set[TimeFrame],
            ) -> None:
                if self._candle_index == 0:
                    bookkeeper.add(
                        account.long_entry(symbol, 1.0).stop_loss(90.0).build()
                    )
                seen_by_candle_index[self._candle_index] = snapshots.of(
                    account
                ).stop_loss_fills(symbol)
                self._candle_index += 1

        ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
        ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
            {
                "open": [100.0] * 6,
                "high": [100.0] * 6,
                "low": [100.0, 80.0, 100.0, 100.0, 100.0, 100.0],
                "close": [100.0] * 6,
                "volume": [1.0] * 6,
            },
            index=pd.date_range("2024-01-01", periods=6, freq="1D", tz="UTC"),
        )
        backtester = Backtester(
            Strategy(),
            simulation_engine,
            ohlcv_provider,
            fill_recorder,
            "2024-01-01",
            "2024-01-06",
        )

        asyncio.run(backtester.run())

        assert seen_by_candle_index[3] != []
        assert seen_by_candle_index[4] == []


class TestReportingQuantityNoTrackingIdOwns:
    def test_a_fill_no_tracking_id_owns_is_warned_about_on_its_candle_alone(
        self, simulation_engine, fill_recorder, account, caplog
    ):
        symbol = Symbol.create("BTC/USDT:USDT")
        tracking_id = TrackingId("BTC/USDT:USDT@1d#alpha")

        class Strategy(StrategyProtocol):
            market_type = "futures"

            def __init__(self) -> None:
                self._entered = False

            async def setup(self, requirements) -> None:
                requirements.ohlcv.add(symbol, "1d", 10)
                requirements.account.add(account, symbols=[symbol])
                requirements.tracker.add(
                    account=account,
                    symbols={tracking_id: symbol},
                    tags={tracking_id: profile_tag("1d", "alpha")},
                )

            def generate_trading_signals(self, ohlcvs: OHLCVs) -> None:
                pass

            def book_trading_actions(
                self,
                _ohlcvs: OHLCVs,
                _snapshots: AccountSnapshots,
                _timestamp: datetime,
                bookkeeper: BookKeeper,
                _triggered_timeframes: set[TimeFrame],
            ) -> None:
                if not self._entered:
                    bookkeeper.add(account.long_entry(symbol, 1.0).build())
                    self._entered = True

        ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
        ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
            {
                "open": [100.0] * 6,
                "high": [100.0] * 6,
                "low": [100.0] * 6,
                "close": [100.0] * 6,
                "volume": [1.0] * 6,
            },
            index=pd.date_range("2024-01-01", periods=6, freq="1D", tz="UTC"),
        )
        backtester = Backtester(
            Strategy(),
            simulation_engine,
            ohlcv_provider,
            fill_recorder,
            "2024-01-01",
            "2024-01-06",
        )

        with caplog.at_level(logging.INFO):
            asyncio.run(backtester.run())

        unowned = [
            record
            for record in caplog.records
            if "of the venue's position" in record.getMessage()
        ]
        assert [record.levelname for record in unowned] == [
            "WARNING",
            "INFO",
            "INFO",
            "INFO",
            "INFO",
        ]


class TestExecutionsWindowSeconds:
    def test_single_requirement(self):
        btc = Symbol.create("BTC/USDT:USDT")
        requirements = [OHLCVRequirement(symbol=btc, timeframe="1d", lookback=20)]

        window_seconds = _executions_window_seconds(requirements)

        assert window_seconds == 19 * 86_400

    def test_multiple_requirements_takes_the_largest(self):
        btc = Symbol.create("BTC/USDT:USDT")
        eth = Symbol.create("ETH/USDT:USDT")
        requirements = [
            OHLCVRequirement(symbol=btc, timeframe="1d", lookback=10),
            OHLCVRequirement(symbol=eth, timeframe="1h", lookback=50),
        ]

        window_seconds = _executions_window_seconds(requirements)

        assert window_seconds == max(9 * 86_400, 49 * 3_600)

    def test_zero_lookback_contributes_no_window(self):
        btc = Symbol.create("BTC/USDT:USDT")
        requirements = [OHLCVRequirement(symbol=btc, timeframe="1d", lookback=0)]

        window_seconds = _executions_window_seconds(requirements)

        assert window_seconds == 0

    def test_empty_requirements(self):
        window_seconds = _executions_window_seconds([])

        assert window_seconds == 0


class TestEquityCurveBounds:
    @pytest.fixture
    def idle_backtest(
        self,
        simulation_engine: FuturesSimulationEngine,
        fill_recorder: CacheFillRecorder,
    ) -> Backtester:
        symbol = Symbol.create("BTC/USDT:USDT")

        class Strategy(StrategyProtocol):
            market_type = "futures"

            async def setup(self, requirements) -> None:
                requirements.ohlcv.add(symbol, "15m")

            def generate_trading_signals(self, ohlcvs: OHLCVs) -> None:
                pass

            def book_trading_actions(self, *_args) -> None:
                pass

        ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
        ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
            {
                "open": [100.0] * 97,
                "high": [100.0] * 97,
                "low": [100.0] * 97,
                "close": [100.0] * 97,
                "volume": [1.0] * 97,
            },
            index=pd.date_range("2024-01-01", periods=97, freq="15min", tz="UTC"),
        )
        return Backtester(
            Strategy(),
            simulation_engine,
            ohlcv_provider,
            fill_recorder,
            "2024-01-01",
            "2024-01-02",
        )

    def test_the_curve_opens_on_the_start_with_the_initial_balance(self, idle_backtest):
        curve = asyncio.run(idle_backtest.run()).get_equity_curve()

        assert curve.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
        assert curve.iloc[0] == 100_000.0

    def test_the_curve_closes_on_the_last_candle_whatever_its_moment(
        self, idle_backtest
    ):
        curve = asyncio.run(idle_backtest.run()).get_equity_curve()

        assert list(curve.index) == [
            pd.Timestamp("2024-01-01 00:00", tz="UTC"),
            pd.Timestamp("2024-01-02 00:00", tz="UTC"),
            pd.Timestamp("2024-01-02 00:15", tz="UTC"),
        ]
