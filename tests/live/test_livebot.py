import asyncio
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab._core import OHLCVProviderProtocol, TimeframeSnapshot
from robottraderslab.exceptions import (
    DataError,
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ExchangeTransientError,
    MissingOhlcvDataError,
    StrategyCriticalError,
)
from robottraderslab.exchanges import (
    Execution,
    FuturesExchangeProtocol,
    MarginMode,
    MarginSettings,
    OrderFill,
    OrderSide,
    OrderType,
    PlacedOrder,
    PositionSide,
    PositionSnapshot,
    VenueFill,
)
from robottraderslab.futures import FuturesAccount
from robottraderslab.live.livebot import LiveBot
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    FillDescriber,
    OHLCVs,
    OrderProtocol,
    ProfileStrategy,
    StrategyProtocol,
    StrategyRequirements,
    TrackingId,
    TradingMode,
    TradingSystem,
    profile_tag,
)


@dataclass(frozen=True, slots=True)
class _Profile:
    profile_id: str
    timeframe: TimeFrame


@pytest.fixture(autouse=True)
def candle_boundary_clock():
    with patch("robottraderslab.live.livebot.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2026, 1, 1, tzinfo=UTC)
        yield mock_datetime


@pytest.fixture
def exchange() -> FuturesExchangeProtocol:
    exchange = Mock(spec=FuturesExchangeProtocol)
    exchange.get_open_positions.return_value = {}
    return exchange


@pytest.fixture
def account(exchange) -> FuturesAccount:
    return FuturesAccount(exchange)


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
                requirements.account.add(account, positions=True)

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


def test_calls_setup_before_generate_trading_signals(account):
    symbol = Symbol.create("BTC/USDT:USDT")
    calls: list[str] = []

    async def add_requirements(requirements):
        calls.append("setup")
        requirements.ohlcv.add(symbol, "1d")

    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)
    strategy.generate_trading_signals.side_effect = lambda ohlcvs: calls.append(
        "generate"
    )

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1000],
        },
        index=[pd.Timestamp("2024-01-01")],
    )
    livebot = LiveBot(strategy, ohlcv_provider)

    asyncio.run(livebot.run())

    assert calls[:2] == ["setup", "generate"]


def test_cycle_logs_the_accounts_it_trades_for_real(account, caplog):
    symbol = Symbol.create("BTC/USDT:USDT")

    async def add_requirements(requirements):
        requirements.ohlcv.add(symbol, "1d")
        requirements.account.add(account, positions=True)

    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1000],
        },
        index=[pd.Timestamp("2024-01-01")],
    )
    livebot = LiveBot(strategy, ohlcv_provider)

    with caplog.at_level(logging.INFO):
        asyncio.run(livebot.run())

    accounts_record = next(
        record
        for record in caplog.records
        if record.getMessage().startswith("Trading real accounts:")
    )
    assert account.name in accounts_record.getMessage()


def test_cycle_logs_the_profile_strategy_once_when_setup_is_retried(caplog):
    symbol = Symbol.create("BTC/USDT:USDT")

    class Strategy(ProfileStrategy[_Profile]):
        market_type = "futures"

        def __init__(self):
            super().__init__(
                account=object(),
                trading_system=TradingSystem(trading_mode=TradingMode.LIVE),
                config_dir=None,
                profiles=[{"profile_id": "alpha", "timeframe": "1d"}],
            )
            self._setup_calls = 0

        async def setup(self, requirements):
            self._setup_calls += 1
            if self._setup_calls == 1:
                raise ExchangeTransientError("502 Bad Gateway")
            requirements.ohlcv.add(symbol, "1d")

        def book_profile_actions(
            self, profile, ohlcvs, account_snapshots, timestamp, bookkeeper
        ):
            pass

    strategy = Strategy()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])
    livebot = LiveBot(strategy, ohlcv_provider, base_delay=0)

    with caplog.at_level(logging.INFO):
        asyncio.run(livebot.run())

    assert strategy._setup_calls == 2
    assert caplog.messages.count("Strategy: 1 profile; slots: per-profile booking") == 1


def test_calls_on_filled_callbacks(
    account, exchange, create_strategy_place_market_order
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
        index=[pd.Timestamp("2024-01-01")],
    )
    livebot = LiveBot(strategy, ohlcv_provider)

    plugin_fill = VenueFill(
        order_id="my-test-id", symbol=symbol, side=OrderSide.BUY, quantity=4.0
    )
    placed_order = PlacedOrder(
        order_id="my-test-id", get_fill=AsyncMock(return_value=plugin_fill)
    )
    exchange.place_market_order.return_value = placed_order

    asyncio.run(livebot.run())

    on_filled.assert_awaited_once_with(
        placed_order,
        OrderFill(
            order_id="my-test-id",
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=4.0,
            kind="market",
            source="strategy",
        ),
    )


@pytest.mark.parametrize(
    "error",
    [
        ExchangeCriticalError("API down"),
        StrategyCriticalError("API down"),
    ],
    ids=["exchange_critical", "strategy_critical"],
)
def test_with_critical_error_in_action_generation(account, error):
    symbol = Symbol.create("BTC/USDT:USDT")

    async def add_requirements(requirements):
        requirements.ohlcv.add(symbol, "1d")

    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)
    strategy.book_trading_actions.side_effect = error

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1000],
        },
        index=[pd.Timestamp("2024-01-01")],
    )
    livebot = LiveBot(strategy, ohlcv_provider)

    with pytest.raises(type(error), match="API down"):
        asyncio.run(livebot.run())


@pytest.mark.parametrize(
    "error",
    [
        ExchangeCriticalError("Auth expired"),
        StrategyCriticalError("Auth expired"),
    ],
    ids=["exchange_critical", "strategy_critical"],
)
def test_with_critical_error_in_callback(
    account, exchange, create_strategy_place_market_order, error
):
    symbol = Symbol.create("BTC/USDT:USDT")
    on_filled = AsyncMock(side_effect=error)
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
        index=[pd.Timestamp("2024-01-01")],
    )
    livebot = LiveBot(strategy, ohlcv_provider)

    order_fill = VenueFill(
        order_id="my-test-id", symbol=symbol, side=OrderSide.BUY, quantity=4.0
    )
    placed_order = PlacedOrder(
        order_id="my-test-id", get_fill=AsyncMock(return_value=order_fill)
    )
    exchange.place_market_order.return_value = placed_order

    with pytest.raises(type(error), match="Auth expired"):
        asyncio.run(livebot.run())


@pytest.mark.parametrize(
    "error",
    [
        ExchangeRecoverableError("Rate limited"),
        RuntimeError("Strategy bug"),
    ],
    ids=["recoverable_error", "generic_error"],
)
def test_non_critical_error_in_action_generation_continues_execution(account, error):
    symbol = Symbol.create("BTC/USDT:USDT")

    async def add_requirements(requirements):
        requirements.ohlcv.add(symbol, "1d")

    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)
    strategy.book_trading_actions.side_effect = error

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1000],
        },
        index=[pd.Timestamp("2024-01-01")],
    )
    livebot = LiveBot(strategy, ohlcv_provider)

    asyncio.run(livebot.run())


@pytest.mark.parametrize(
    ("error", "reported"),
    [
        (
            ExchangeRecoverableError("Timeout"),
            "Exchange rejected operation in callback",
        ),
        (RuntimeError("Callback bug"), "Internal error in callback"),
        (
            asyncio.CancelledError("Callback cancelled itself"),
            "Internal error in callback",
        ),
    ],
    ids=["recoverable_error", "generic_error", "base_exception"],
)
def test_non_critical_error_in_callback_continues_execution(
    account, exchange, create_strategy_place_market_order, caplog, error, reported
):
    symbol = Symbol.create("BTC/USDT:USDT")
    on_filled = AsyncMock(side_effect=error)
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
        index=[pd.Timestamp("2024-01-01")],
    )
    livebot = LiveBot(strategy, ohlcv_provider)

    order_fill = VenueFill(
        order_id="my-test-id", symbol=symbol, side=OrderSide.BUY, quantity=4.0
    )
    placed_order = PlacedOrder(
        order_id="my-test-id", get_fill=AsyncMock(return_value=order_fill)
    )
    exchange.place_market_order.return_value = placed_order

    with caplog.at_level(logging.WARNING):
        asyncio.run(livebot.run())

    assert reported in caplog.text


def test_declared_account_state_reaches_the_strategy(account, exchange):
    symbol = Symbol.create("BTC/USDT:USDT")
    position = Mock()
    exchange.get_open_positions.return_value = {symbol: position}
    observed = {}

    class Strategy(StrategyProtocol):
        market_type = "futures"

        async def setup(self, requirements):
            requirements.ohlcv.add(symbol, "1d")
            requirements.account.add(account, positions=True)

        def generate_trading_signals(self, ohlcvs):
            pass

        def book_trading_actions(
            self, ohlcvs, account_snapshots, timestamp, bookkeeper, triggered_timeframes
        ):
            observed["position"] = account_snapshots.of(account).position(symbol)

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1000],
        },
        index=[pd.Timestamp("2024-01-01")],
    )
    livebot = LiveBot(Strategy(), ohlcv_provider)

    asyncio.run(livebot.run())

    assert observed["position"] is position
    exchange.get_open_positions.assert_awaited_once()


def test_no_account_reads_without_state_declarations(account, exchange):
    symbol = Symbol.create("BTC/USDT:USDT")

    async def add_requirements(requirements):
        requirements.ohlcv.add(symbol, "1d")

    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1000],
        },
        index=[pd.Timestamp("2024-01-01")],
    )
    livebot = LiveBot(strategy, ohlcv_provider)

    asyncio.run(livebot.run())

    exchange.get_open_positions.assert_not_awaited()
    exchange.get_open_orders.assert_not_awaited()
    exchange.get_balances.assert_not_awaited()


def test_raises_when_no_ohlcv_requirements(account):
    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock()

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    livebot = LiveBot(strategy, ohlcv_provider)

    with pytest.raises(StrategyCriticalError, match="declared no OHLCV requirements"):
        asyncio.run(livebot.run())


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
def test_ohlcv_fetch_with_transient_error_then_success(mock_sleep, account):
    symbol = Symbol.create("BTC/USDT:USDT")

    async def add_requirements(requirements):
        requirements.ohlcv.add(symbol, "1d")

    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)

    valid_df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1000],
        },
        index=[pd.Timestamp("2024-01-01")],
    )
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.side_effect = [
        ExchangeTransientError("502 Bad Gateway"),
        valid_df,
    ]
    livebot = LiveBot(strategy, ohlcv_provider)

    asyncio.run(livebot.run())

    assert ohlcv_provider.fetch_ohlcv.call_count == 2
    strategy.generate_trading_signals.assert_called_once()


def _candles(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0] * len(timestamps),
            "high": [110.0] * len(timestamps),
            "low": [90.0] * len(timestamps),
            "close": [105.0] * len(timestamps),
            "volume": [1000.0] * len(timestamps),
        },
        index=pd.to_datetime(timestamps),
    )


@pytest.fixture
def create_recording_strategy():
    def _create_recording_strategy(
        symbol: Symbol,
        timeframes: list[TimeFrame],
        observed_bookings: list[tuple[datetime, set[TimeFrame]]],
    ) -> StrategyProtocol:
        class Strategy(StrategyProtocol):
            market_type = "futures"

            async def setup(self, requirements):
                for timeframe in timeframes:
                    requirements.ohlcv.add(symbol, timeframe)

            def generate_trading_signals(self, ohlcvs):
                pass

            def book_trading_actions(
                self, ohlcvs, snapshots, timestamp, bookkeeper, triggered_timeframes
            ):
                observed_bookings.append((timestamp, set(triggered_timeframes)))

        return Strategy()

    return _create_recording_strategy


def test_skips_the_run_off_candle_boundaries(
    create_recording_strategy, candle_boundary_clock
):
    symbol = Symbol.create("BTC/USDT:USDT")
    observed_bookings: list[tuple[datetime, set[TimeFrame]]] = []
    strategy = create_recording_strategy(symbol, ["1d"], observed_bookings)
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    candle_boundary_clock.now.return_value = datetime(2026, 1, 1, 3, 7, tzinfo=UTC)

    livebot = LiveBot(strategy, ohlcv_provider)
    asyncio.run(livebot.run())

    ohlcv_provider.fetch_ohlcv.assert_not_called()
    assert observed_bookings == []


def test_books_only_the_timeframes_on_their_boundary(
    create_recording_strategy, candle_boundary_clock
):
    symbol = Symbol.create("BTC/USDT:USDT")
    observed_bookings: list[tuple[datetime, set[TimeFrame]]] = []
    strategy = create_recording_strategy(symbol, ["5m", "15m"], observed_bookings)
    frames = {
        "5m": _candles(["2026-01-01 08:55", "2026-01-01 09:00"]),
        "15m": _candles(["2026-01-01 08:30", "2026-01-01 08:45"]),
    }
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.side_effect = lambda _, timeframe: frames[timeframe]
    candle_boundary_clock.now.return_value = datetime(2026, 1, 1, 9, 5, 1, tzinfo=UTC)

    livebot = LiveBot(strategy, ohlcv_provider)
    asyncio.run(livebot.run())

    assert observed_bookings == [(pd.Timestamp("2026-01-01 09:05"), {"5m"})]


def test_a_shared_close_books_once_with_both_timeframes(create_recording_strategy):
    symbol = Symbol.create("BTC/USDT:USDT")
    observed_bookings: list[tuple[datetime, set[TimeFrame]]] = []
    strategy = create_recording_strategy(symbol, ["5m", "15m"], observed_bookings)
    frames = {
        "5m": _candles(["2026-01-01 09:45", "2026-01-01 09:50", "2026-01-01 09:55"]),
        "15m": _candles(["2026-01-01 09:30", "2026-01-01 09:45"]),
    }
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.side_effect = lambda _, timeframe: frames[timeframe]

    livebot = LiveBot(strategy, ohlcv_provider)
    asyncio.run(livebot.run())

    assert observed_bookings == [(pd.Timestamp("2026-01-01 10:00"), {"5m", "15m"})]


def test_a_closing_timeframe_whose_candle_was_not_fetched_is_not_booked(
    create_recording_strategy, caplog
):
    symbol = Symbol.create("BTC/USDT:USDT")
    observed_bookings: list[tuple[datetime, set[TimeFrame]]] = []
    strategy = create_recording_strategy(symbol, ["5m", "15m"], observed_bookings)
    frames = {
        "5m": _candles(["2026-01-01 09:45", "2026-01-01 09:50", "2026-01-01 09:55"]),
        "15m": _candles(["2026-01-01 09:15", "2026-01-01 09:30"]),
    }
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.side_effect = lambda _, timeframe: frames[timeframe]

    livebot = LiveBot(strategy, ohlcv_provider)
    with caplog.at_level(logging.WARNING):
        asyncio.run(livebot.run())

    assert observed_bookings == [(pd.Timestamp("2026-01-01 10:00"), {"5m"})]
    assert (
        "No candle closing at 2026-01-01 10:00:00 was fetched for 15m, not booked"
        in caplog.messages
    )


def test_no_booking_when_no_closing_timeframe_reaches_the_newest_close(
    create_recording_strategy, candle_boundary_clock
):
    symbol = Symbol.create("BTC/USDT:USDT")
    observed_bookings: list[tuple[datetime, set[TimeFrame]]] = []
    strategy = create_recording_strategy(symbol, ["5m", "1h"], observed_bookings)
    frames = {
        "5m": _candles(["2026-01-01 09:45", "2026-01-01 09:50"]),
        "1h": _candles(["2026-01-01 08:00", "2026-01-01 09:00"]),
    }
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.side_effect = lambda _, timeframe: frames[timeframe]
    candle_boundary_clock.now.return_value = datetime(2026, 1, 1, 10, 5, 1, tzinfo=UTC)

    livebot = LiveBot(strategy, ohlcv_provider)
    asyncio.run(livebot.run())

    assert observed_bookings == []


def test_notifier_fires_only_after_every_action_of_the_cycle_is_placed(
    account, exchange
):
    symbol = Symbol.create("BTC/USDT:USDT")
    log: list[str] = []

    class Strategy(StrategyProtocol):
        market_type = "futures"

        async def setup(self, requirements):
            requirements.ohlcv.add(symbol, "5m")
            requirements.ohlcv.add(symbol, "15m")

        def generate_trading_signals(self, ohlcvs):
            pass

        def book_trading_actions(
            self, ohlcvs, snapshots, timestamp, bookkeeper, triggered_timeframes
        ):
            bookkeeper.add(account.long_entry(symbol, 1.0).build())
            bookkeeper.add(account.long_entry(symbol, 2.0).build())

    def _place_market_order(**_kwargs):
        log.append("placed")
        order_id = f"order-{len(log)}"
        order_fill = VenueFill(
            order_id=order_id, symbol=symbol, side=OrderSide.BUY, quantity=1.0
        )
        return PlacedOrder(
            order_id=order_id,
            get_fill=AsyncMock(return_value=order_fill),
        )

    exchange.place_market_order = AsyncMock(side_effect=_place_market_order)

    frames = {
        "5m": _candles(["2026-01-01 09:45", "2026-01-01 09:50", "2026-01-01 09:55"]),
        "15m": _candles(["2026-01-01 09:30", "2026-01-01 09:45"]),
    }
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.side_effect = lambda _, timeframe: frames[timeframe]

    fill_notifier = AsyncMock(side_effect=lambda _order: log.append("notified"))
    placement_notifier = AsyncMock(side_effect=lambda _order: log.append("announced"))
    livebot = LiveBot(
        Strategy(),
        ohlcv_provider,
        on_fill=[fill_notifier],
        on_placement=[placement_notifier],
    )

    asyncio.run(livebot.run())

    assert log[:2] == ["placed", "placed"]
    assert sorted(log[2:]) == ["announced", "announced", "notified", "notified"]


@pytest.fixture
def create_entry_fill_strategy(account):
    def _create_entry_fill_strategy(
        symbol: Symbol,
        *,
        describer: FillDescriber | None = None,
        symbols: Iterable[Symbol] = (),
    ) -> StrategyProtocol:
        class Strategy(StrategyProtocol):
            market_type = "futures"

            async def setup(self, requirements):
                requirements.ohlcv.add(symbol, "1d")
                requirements.account.add(
                    account,
                    symbols=symbols,
                    notify_entry_fills=True,
                    describe_fills=describer,
                )

            def generate_trading_signals(self, ohlcvs):
                pass

            def book_trading_actions(
                self, ohlcvs, snapshots, timestamp, bookkeeper, triggered_timeframes
            ):
                pass

        return Strategy()

    return _create_entry_fill_strategy


def _entry_fill(
    symbol: Symbol, timestamp: datetime, *, kind: OrderType = "trigger"
) -> Execution:
    return Execution(
        execution_id="fired-1",
        order_id="fired-1",
        symbol=symbol,
        side=OrderSide.BUY,
        price=95.0,
        quantity=1.0,
        timestamp=timestamp,
        kind=kind,
        client_order_id="fakeprocess99-1-r1",
    )


@pytest.mark.parametrize("kind", ["trigger", "limit"])
def test_entry_fills_of_the_closed_candle_reach_notifiers(
    exchange, create_entry_fill_strategy, caplog, kind
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[
            _entry_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC), kind=kind)
        ]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_entry_fill_strategy(
        symbol, describer=lambda fill: f"entry on {fill.symbol}"
    )
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    with caplog.at_level(logging.INFO):
        asyncio.run(livebot.run())

    notifier.assert_awaited_once()
    notified_order = notifier.await_args.args[0]
    assert notified_order.order_id == "fired-1"
    assert notified_order.kind == kind
    assert notified_order.reason == f"entry on {symbol}"
    assert "fired-1" in caplog.text


def test_a_fill_of_no_entry_kind_is_not_notified_as_an_entry(
    exchange, create_entry_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[
            _entry_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC), kind="stop-loss")
        ]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    livebot = LiveBot(
        create_entry_fill_strategy(symbol), ohlcv_provider, on_fill=[notifier]
    )
    asyncio.run(livebot.run())

    notifier.assert_not_awaited()


def test_declared_symbols_scope_the_entry_fill_read(
    exchange, create_entry_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(return_value=[])
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_entry_fill_strategy(symbol, symbols=[symbol])
    livebot = LiveBot(strategy, ohlcv_provider)

    asyncio.run(livebot.run())

    assert exchange.get_executions_since.await_args.args[1] == [symbol]


def test_entry_fills_outside_the_closed_candle_are_left_alone(
    exchange, create_entry_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[_entry_fill(symbol, datetime(2024, 1, 2, 1, tzinfo=UTC))]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    livebot = LiveBot(
        create_entry_fill_strategy(symbol),
        ohlcv_provider,
        on_fill=[notifier],
    )
    asyncio.run(livebot.run())

    notifier.assert_not_awaited()


def test_entry_fills_are_notified_without_a_reason_when_no_describer_is_declared(
    exchange, create_entry_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[_entry_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC))]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    livebot = LiveBot(
        create_entry_fill_strategy(symbol),
        ohlcv_provider,
        on_fill=[notifier],
    )
    asyncio.run(livebot.run())

    notifier.assert_awaited_once()
    assert notifier.await_args.args[0].reason is None


def test_entry_fills_the_describer_disowns_are_left_alone(
    exchange, create_entry_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[_entry_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC))]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_entry_fill_strategy(symbol, describer=lambda fill: None)
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())

    notifier.assert_not_awaited()


def test_rejected_entry_fill_read_warns_and_finishes_the_cycle(
    exchange, create_entry_fill_strategy, caplog
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        side_effect=ExchangeRecoverableError("read rejected")
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    livebot = LiveBot(
        create_entry_fill_strategy(symbol),
        ohlcv_provider,
        on_fill=[notifier],
    )
    with caplog.at_level(logging.WARNING):
        asyncio.run(livebot.run())

    assert "Entry fill reconciliation failed" in caplog.text
    notifier.assert_not_awaited()


def test_no_entry_fill_reads_without_the_declaration(exchange, account):
    symbol = Symbol.create("BTC/USDT:USDT")

    async def add_requirements(requirements):
        requirements.ohlcv.add(symbol, "1d")
        requirements.account.add(account, positions=True)

    exchange.get_open_positions.return_value = {}
    exchange.get_executions_since = AsyncMock(return_value=[])
    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    livebot = LiveBot(strategy, ohlcv_provider)
    asyncio.run(livebot.run())

    exchange.get_executions_since.assert_not_awaited()


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
def test_ohlcv_fetch_with_persistent_transient_error(mock_sleep, account, caplog):
    symbol = Symbol.create("BTC/USDT:USDT")

    async def add_requirements(requirements):
        requirements.ohlcv.add(symbol, "1d")

    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.side_effect = ExchangeTransientError("502 Bad Gateway")
    livebot = LiveBot(strategy, ohlcv_provider)

    asyncio.run(livebot.run())

    strategy.generate_trading_signals.assert_not_called()
    assert "No declared market data could be fetched, skipping cycle" in caplog.text


def test_ohlcv_fetch_with_an_error_a_retry_cannot_fix(account, caplog):
    symbol = Symbol.create("BTC/USDT:USDT")

    async def add_requirements(requirements):
        requirements.ohlcv.add(symbol, "1d")

    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.side_effect = DataError(
        "BTC/USDT:USDT@1d: failed to fetch from exchange: not listed"
    )
    livebot = LiveBot(strategy, ohlcv_provider)

    asyncio.run(livebot.run())

    strategy.generate_trading_signals.assert_not_called()
    assert ohlcv_provider.fetch_ohlcv.call_count == 1
    assert "No declared market data could be fetched, skipping cycle" in caplog.text


def test_one_failing_symbol_does_not_block_the_others(account, caplog):
    good = Symbol.create("BTC/USDT:USDT")
    bad = Symbol.create("ETH/USDT:USDT")

    async def add_requirements(requirements):
        requirements.ohlcv.add(good, "1d")
        requirements.ohlcv.add(bad, "1d")

    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)

    def fetch_ohlcv(symbol, timeframe):
        if symbol == bad:
            raise DataError(f"{symbol}@{timeframe}: not listed")
        return _candles(["2024-01-01"])

    ohlcv_provider.fetch_ohlcv.side_effect = fetch_ohlcv
    livebot = LiveBot(strategy, ohlcv_provider)

    asyncio.run(livebot.run())

    strategy.generate_trading_signals.assert_called_once()
    ohlcvs = strategy.generate_trading_signals.call_args.args[0]
    assert len(ohlcvs.column(good, "1d", "close")) == 1
    with pytest.raises(MissingOhlcvDataError):
        ohlcvs.column(bad, "1d", "close")
    assert "skipping this cycle" in caplog.text


def test_a_strategy_error_in_signal_generation_skips_the_cycle(account, caplog):
    symbol = Symbol.create("BTC/USDT:USDT")

    async def add_requirements(requirements):
        requirements.ohlcv.add(symbol, "1d")

    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)
    strategy.generate_trading_signals.side_effect = KeyError("missing column")

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])
    livebot = LiveBot(strategy, ohlcv_provider)

    asyncio.run(livebot.run())

    strategy.book_trading_actions.assert_not_called()
    assert "Signal generation failed, skipping cycle" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        ExchangeCriticalError("Signals broke"),
        StrategyCriticalError("Signals broke"),
    ],
    ids=["exchange_critical", "strategy_critical"],
)
def test_with_critical_error_in_signal_generation(account, error):
    symbol = Symbol.create("BTC/USDT:USDT")

    async def add_requirements(requirements):
        requirements.ohlcv.add(symbol, "1d")

    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)
    strategy.generate_trading_signals.side_effect = error

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])
    livebot = LiveBot(strategy, ohlcv_provider)

    with pytest.raises(type(error), match="Signals broke"):
        asyncio.run(livebot.run())


@pytest.mark.parametrize(
    "error",
    [
        ExchangeCriticalError("Refresh broke"),
        StrategyCriticalError("Refresh broke"),
    ],
    ids=["exchange_critical", "strategy_critical"],
)
def test_with_critical_error_in_tracker_refresh(account, error):
    symbol = Symbol.create("BTC/USDT:USDT")

    async def add_requirements(requirements):
        requirements.ohlcv.add(symbol, "1d")
        requirements.account.add(account, symbols=[symbol])

    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)

    requirements = StrategyRequirements()
    requirements.tracker.refresh_all = Mock(side_effect=error)

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])
    livebot = LiveBot(
        strategy, ohlcv_provider, requirements_factory=lambda: requirements
    )

    with pytest.raises(type(error), match="Refresh broke"):
        asyncio.run(livebot.run())


@pytest.fixture
def create_tracking_strategy(account):
    def _create_tracking_strategy(symbol: Symbol) -> StrategyProtocol:
        tracking_id = TrackingId(f"{symbol}@1d#alpha")

        class Strategy(StrategyProtocol):
            market_type = "futures"

            async def setup(self, requirements):
                requirements.ohlcv.add(symbol, "1d")
                requirements.account.add(account, symbols=[symbol])
                requirements.tracker.add(
                    account=account,
                    symbols={tracking_id: symbol},
                    tags={tracking_id: profile_tag("1d", "alpha")},
                )

            def generate_trading_signals(self, ohlcvs):
                pass

            def book_trading_actions(
                self, ohlcvs, snapshots, timestamp, bookkeeper, triggered_timeframes
            ):
                pass

        return Strategy()

    return _create_tracking_strategy


def test_a_hand_on_the_venue_inside_the_closed_candle_is_warned_about(
    exchange, create_tracking_strategy, caplog
):
    symbol = Symbol.create("BTC/USDT:USDT")
    _hold_a_position_opened_by_hand(
        exchange, symbol, datetime(2024, 1, 1, 5, tzinfo=UTC)
    )
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(
        ["2023-12-30", "2023-12-31", "2024-01-01"]
    )

    with caplog.at_level(logging.INFO):
        asyncio.run(LiveBot(create_tracking_strategy(symbol), ohlcv_provider).run())

    assert _messages_at(caplog, logging.WARNING) == [
        "BTC/USDT:USDT: +1 of the venue's position was opened by an order this "
        "strategy did not place, on 2024-01-01 05:00 UTC"
    ]


def test_a_hand_on_the_venue_before_the_closed_candle_is_only_stated(
    exchange, create_tracking_strategy, caplog
):
    symbol = Symbol.create("BTC/USDT:USDT")
    _hold_a_position_opened_by_hand(
        exchange, symbol, datetime(2023, 12, 30, 5, tzinfo=UTC)
    )
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(
        ["2023-12-30", "2023-12-31", "2024-01-01"]
    )

    with caplog.at_level(logging.INFO):
        asyncio.run(LiveBot(create_tracking_strategy(symbol), ohlcv_provider).run())

    assert _messages_at(caplog, logging.WARNING) == []
    assert (
        "BTC/USDT:USDT: +1 of the venue's position is held outside this strategy"
        in _messages_at(caplog, logging.INFO)
    )


def _hold_a_position_opened_by_hand(
    exchange, symbol: Symbol, timestamp: datetime
) -> None:
    exchange.get_open_positions.return_value = {
        symbol: PositionSnapshot(
            symbol=symbol,
            side=PositionSide.LONG,
            quantity=1.0,
            average_entry_price=100.0,
            entry_time=timestamp,
            leverage=1.0,
            liquidation_price=0.0,
        )
    }
    exchange.get_executions_since = AsyncMock(
        return_value=[
            Execution(
                execution_id="by-hand-1",
                order_id="by-hand-1",
                symbol=symbol,
                side=OrderSide.BUY,
                price=100.0,
                quantity=1.0,
                timestamp=timestamp,
            )
        ]
    )


def _messages_at(caplog, level: int) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == level]


@pytest.fixture
def create_venue_fill_strategy(account) -> Callable[..., StrategyProtocol]:
    def _create_venue_fill_strategy(
        symbol: Symbol, *, describer: FillDescriber | None = None
    ) -> StrategyProtocol:
        class Strategy(StrategyProtocol):
            market_type = "futures"

            async def setup(self, requirements):
                requirements.ohlcv.add(symbol, "1d")
                requirements.account.add(
                    account,
                    symbols=[symbol],
                    executions=True,
                    describe_fills=describer,
                )

            def generate_trading_signals(self, ohlcvs):
                pass

            def book_trading_actions(
                self, ohlcvs, snapshots, timestamp, bookkeeper, triggered_timeframes
            ):
                pass

        return Strategy()

    return _create_venue_fill_strategy


@pytest.fixture
def create_undeclared_fill_strategy(account) -> Callable[[Symbol], StrategyProtocol]:
    def _create_undeclared_fill_strategy(symbol: Symbol) -> StrategyProtocol:
        class Strategy(StrategyProtocol):
            market_type = "futures"

            async def setup(self, requirements):
                requirements.ohlcv.add(symbol, "1d")
                requirements.account.add(account, symbols=[symbol])

            def generate_trading_signals(self, ohlcvs):
                pass

            def book_trading_actions(
                self, ohlcvs, snapshots, timestamp, bookkeeper, triggered_timeframes
            ):
                pass

        return Strategy()

    return _create_undeclared_fill_strategy


def _stop_loss_fill(symbol: Symbol, timestamp: datetime) -> Execution:
    return Execution(
        execution_id="sl-1",
        order_id="sl-1",
        symbol=symbol,
        side=OrderSide.SELL,
        price=100.0,
        quantity=1.0,
        timestamp=timestamp,
        kind="stop-loss",
    )


def _take_profit_fill(symbol: Symbol, timestamp: datetime) -> Execution:
    return Execution(
        execution_id="tp-1",
        order_id="tp-1",
        symbol=symbol,
        side=OrderSide.BUY,
        price=100.0,
        quantity=1.0,
        timestamp=timestamp,
        kind="take-profit",
    )


def _long_position(symbol: Symbol, quantity: float) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        side=PositionSide.LONG,
        quantity=quantity,
        average_entry_price=100.0,
        entry_time=datetime(2024, 1, 1, tzinfo=UTC),
        leverage=1.0,
        liquidation_price=1.0,
    )


def _liquidation_fill(symbol: Symbol, timestamp: datetime) -> Execution:
    return Execution(
        execution_id="liq-1",
        order_id="liq-1",
        symbol=symbol,
        side=OrderSide.SELL,
        price=100.0,
        quantity=1.0,
        timestamp=timestamp,
        kind="liquidation",
    )


def test_stop_loss_fill_of_the_closed_candle_reaches_notifiers(
    exchange, create_venue_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[_stop_loss_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC))]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_venue_fill_strategy(symbol)
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())

    notifier.assert_awaited_once()
    notified_order = notifier.await_args.args[0]
    assert notified_order.order_id == "sl-1"
    assert notified_order.kind == "stop-loss"
    assert notified_order.quantity == 1.0


@pytest.mark.parametrize(
    ("create_fill", "kind"),
    [
        (_stop_loss_fill, "stop-loss"),
        (_take_profit_fill, "take-profit"),
        (_liquidation_fill, "liquidation"),
    ],
    ids=["stop_loss", "take_profit", "liquidation"],
)
def test_venue_fired_fill_is_reported_once_under_an_entry_fill_declaration(
    exchange, create_entry_fill_strategy, create_fill, kind
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[create_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC))]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_entry_fill_strategy(symbol, symbols=[symbol])
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())

    notifier.assert_awaited_once()
    assert notifier.await_args.args[0].kind == kind


def test_take_profit_fill_of_the_closed_candle_reaches_notifiers(
    exchange, create_venue_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[_take_profit_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC))]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_venue_fill_strategy(symbol)
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())

    notifier.assert_awaited_once()
    notified_order = notifier.await_args.args[0]
    assert notified_order.order_id == "tp-1"
    assert notified_order.kind == "take-profit"


def test_liquidation_fill_of_the_closed_candle_reaches_notifiers(
    exchange, create_venue_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[_liquidation_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC))]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_venue_fill_strategy(symbol)
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())

    notifier.assert_awaited_once()
    notified_order = notifier.await_args.args[0]
    assert notified_order.order_id == "liq-1"
    assert notified_order.kind == "liquidation"


@pytest.mark.parametrize(
    ("create_fill", "source"),
    [
        (_stop_loss_fill, "stop-loss"),
        (_take_profit_fill, "take-profit"),
        (_liquidation_fill, "liquidation"),
    ],
    ids=["stop_loss", "take_profit", "liquidation"],
)
def test_venue_fired_fill_names_what_fired_it(
    exchange, create_venue_fill_strategy, create_fill, source
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[create_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC))]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_venue_fill_strategy(symbol)
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())

    assert notifier.await_args.args[0].source == source


def test_venue_fired_fill_carries_the_reason_the_strategy_gives(
    exchange, create_venue_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[_take_profit_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC))]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_venue_fill_strategy(
        symbol, describer=lambda fill: f"exit at reference on {fill.kind}"
    )
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())

    assert notifier.await_args.args[0].reason == "exit at reference on take-profit"


def test_venue_fired_fill_carries_the_profit_the_venue_booked(
    exchange, create_venue_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    fill = replace(
        _take_profit_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC)),
        realised_profit=-7.25,
    )
    exchange.get_executions_since = AsyncMock(return_value=[fill])
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_venue_fill_strategy(symbol)
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())

    assert notifier.await_args.args[0].realised_profit == -7.25


def test_venue_fired_fill_states_what_it_did_to_the_position(
    exchange, create_venue_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[_stop_loss_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC))]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_venue_fill_strategy(symbol)
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())

    assert notifier.await_args.args[0].effect == "close"


def test_stop_loss_fill_outside_the_closed_candle_is_left_alone(
    exchange, create_venue_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[_stop_loss_fill(symbol, datetime(2024, 1, 2, 1, tzinfo=UTC))]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_venue_fill_strategy(symbol)
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())

    notifier.assert_not_awaited()


def test_no_tp_sl_fetch_without_the_declaration(exchange, account):
    symbol = Symbol.create("BTC/USDT:USDT")

    async def add_requirements(requirements):
        requirements.ohlcv.add(symbol, "1d")
        requirements.account.add(account, positions=True)

    exchange.get_open_positions.return_value = {}
    exchange.get_executions_since = AsyncMock(return_value=[])
    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    livebot = LiveBot(strategy, ohlcv_provider)
    asyncio.run(livebot.run())

    exchange.get_executions_since.assert_not_awaited()


def test_stop_loss_fill_reaches_a_subscriber_the_strategy_never_declared(
    exchange, create_undeclared_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[_stop_loss_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC))]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_undeclared_fill_strategy(symbol)
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())

    notifier.assert_awaited_once()
    assert notifier.await_args.args[0].order_id == "sl-1"


def test_no_venue_fired_fetch_without_a_subscriber_or_a_declaration(
    exchange, create_undeclared_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(return_value=[])
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_undeclared_fill_strategy(symbol)
    livebot = LiveBot(strategy, ohlcv_provider)
    asyncio.run(livebot.run())

    exchange.get_executions_since.assert_not_awaited()


def test_an_entry_fill_seen_by_two_consecutive_cycles_is_reported_once(
    exchange, create_entry_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[_entry_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC))]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.side_effect = [
        _candles(["2024-01-01"]),
        _candles(["2024-01-01", "2024-01-02"]),
    ]

    strategy = create_entry_fill_strategy(symbol)
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())
    asyncio.run(livebot.run())

    notifier.assert_awaited_once()


def test_a_fill_seen_by_two_consecutive_cycles_is_reported_once(
    exchange, create_venue_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    fill = _stop_loss_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC))
    exchange.get_executions_since = AsyncMock(return_value=[fill])
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.side_effect = [
        _candles(["2024-01-01"]),
        _candles(["2024-01-01", "2024-01-02"]),
    ]

    strategy = create_venue_fill_strategy(symbol)
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())
    asyncio.run(livebot.run())

    notifier.assert_awaited_once()


def test_entry_fill_states_what_the_strategy_did_to_the_position(
    exchange, create_entry_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[_entry_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC))]
    )
    exchange.get_open_positions.return_value = {symbol: _long_position(symbol, 1.0)}
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_entry_fill_strategy(symbol, symbols=[symbol])
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())

    notified_order = notifier.await_args.args[0]
    assert notified_order.effect == "open"
    assert notified_order.source == "strategy"


def test_entry_fill_with_no_stream_to_attribute_it_against(
    exchange, create_entry_fill_strategy
):
    symbol = Symbol.create("BTC/USDT:USDT")
    exchange.get_executions_since = AsyncMock(
        return_value=[_entry_fill(symbol, datetime(2024, 1, 1, 5, tzinfo=UTC))]
    )
    notifier = AsyncMock()
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])

    strategy = create_entry_fill_strategy(symbol)
    livebot = LiveBot(strategy, ohlcv_provider, on_fill=[notifier])
    asyncio.run(livebot.run())

    notified_order = notifier.await_args.args[0]
    assert notified_order.effect is None
    assert notified_order.source == "strategy"


DOGE = Symbol.create("DOGE/USDT:USDT")
TAKE_PROFIT = 0.25
TAKE_PROFIT_ORDER = PlacedOrder(order_id="tp-1")


def _resting_order(symbol: Symbol, kind: str, order_id: str) -> Mock:
    order = Mock(spec=OrderProtocol)
    order.symbol = symbol
    order.kind = kind
    order.order_id = order_id
    return order


@pytest.fixture
def create_protection_strategy(account) -> Callable[[Symbol, float], StrategyProtocol]:
    def _create_protection_strategy(
        symbol: Symbol, take_profit: float
    ) -> StrategyProtocol:
        class Strategy(StrategyProtocol):
            market_type = "futures"

            async def setup(self, requirements):
                requirements.ohlcv.add(symbol, "1d")
                requirements.account.add(account, symbols=[symbol], positions=True)

            def generate_trading_signals(self, ohlcvs):
                pass

            def book_trading_actions(
                self, ohlcvs, snapshots, timestamp, bookkeeper, triggered_timeframes
            ):
                bookkeeper.add(account.move_take_profit(symbol, take_profit))

        return Strategy()

    return _create_protection_strategy


@pytest.fixture
def protection_livebot(exchange, create_protection_strategy) -> LiveBot:
    exchange.update_position_take_profit.return_value = TAKE_PROFIT_ORDER
    exchange.get_open_orders.return_value = [
        _resting_order(DOGE, "take-profit", TAKE_PROFIT_ORDER.order_id)
    ]
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])
    return LiveBot(create_protection_strategy(DOGE, TAKE_PROFIT), ohlcv_provider)


def test_a_protection_the_venue_rejects_once_rests_at_the_booked_level_when_the_cycle_ends(
    exchange, protection_livebot
):
    exchange.update_position_take_profit.side_effect = [
        ExchangeRecoverableError("code 99999"),
        TAKE_PROFIT_ORDER,
    ]

    asyncio.run(protection_livebot.run())

    assert exchange.update_position_take_profit.await_count == 2
    exchange.update_position_take_profit.assert_awaited_with(DOGE, TAKE_PROFIT)


def test_a_protection_the_venue_rejects_twice_names_the_symbol_the_protection_and_the_level(
    exchange, protection_livebot, caplog
):
    exchange.update_position_take_profit.side_effect = ExchangeRecoverableError(
        "code 99999"
    )

    with caplog.at_level(logging.WARNING):
        asyncio.run(protection_livebot.run())

    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert "DOGE/USDT:USDT" in record.message
    assert "take-profit" in record.message
    assert "0.25" in record.message


def test_a_cycle_whose_protections_all_rest_as_booked(
    exchange, protection_livebot, caplog
):
    with caplog.at_level(logging.WARNING):
        asyncio.run(protection_livebot.run())

    exchange.update_position_take_profit.assert_awaited_once_with(DOGE, TAKE_PROFIT)
    assert caplog.records == []


def test_a_cycle_booking_protections_reads_the_venue_back_once_after_its_actions_ran(
    exchange, protection_livebot
):
    asyncio.run(protection_livebot.run())

    exchange.get_open_orders.assert_awaited_once_with([DOGE])
    call_names = [name for name, _args, _kwargs in exchange.mock_calls]
    assert call_names.index("update_position_take_profit") < call_names.index(
        "get_open_orders"
    )


def test_a_cycle_booking_no_protection_reads_nothing_back(account, exchange):
    async def add_requirements(requirements):
        requirements.ohlcv.add(DOGE, "1d")
        requirements.account.add(account, symbols=[DOGE], positions=True)

    strategy = Mock(spec=StrategyProtocol)
    strategy.setup = AsyncMock(side_effect=add_requirements)
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])
    livebot = LiveBot(strategy, ohlcv_provider)

    asyncio.run(livebot.run())

    exchange.get_open_orders.assert_not_awaited()


def test_a_protection_the_venue_holds_under_another_id_is_re_issued_and_read_back(
    exchange, protection_livebot
):
    exchange.get_open_orders.side_effect = [
        [_resting_order(DOGE, "take-profit", "tp-old")],
        [_resting_order(DOGE, "take-profit", TAKE_PROFIT_ORDER.order_id)],
    ]

    asyncio.run(protection_livebot.run())

    assert exchange.update_position_take_profit.await_count == 2
    assert exchange.get_open_orders.await_count == 2


def test_a_protection_absent_after_the_re_issue_names_the_symbol_the_protection_and_the_level(
    exchange, protection_livebot, caplog
):
    exchange.get_open_orders.return_value = []

    with caplog.at_level(logging.WARNING):
        asyncio.run(protection_livebot.run())

    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert "DOGE/USDT:USDT" in record.message
    assert "take-profit" in record.message
    assert "0.25" in record.message


ISOLATED_5X = MarginSettings(leverage=5.0, margin_mode=MarginMode.ISOLATED)


@pytest.fixture
def create_margin_target_livebot(
    account: FuturesAccount, exchange: Mock
) -> Callable[[list[str], float], LiveBot]:
    def _create(events: list[str], held_leverage: float) -> LiveBot:
        exchange.get_margin_settings.return_value = {
            DOGE: MarginSettings(
                leverage=held_leverage, margin_mode=MarginMode.ISOLATED
            )
        }
        exchange.set_leverage.side_effect = lambda *_: events.append("set_leverage")

        class Strategy(StrategyProtocol):
            market_type = "futures"

            async def setup(self, requirements):
                requirements.ohlcv.add(DOGE, "1d")
                requirements.account.add(account, margin_targets={DOGE: ISOLATED_5X})

            def generate_trading_signals(self, ohlcvs):
                pass

            def book_trading_actions(
                self, ohlcvs, account_snapshots, timestamp, bookkeeper, timeframes
            ):
                leverage = account_snapshots.of(account).margin_settings(DOGE).leverage
                events.append(f"book at {leverage}")
                bookkeeper.add(account.long_entry(DOGE, 1.0))

        ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
        ohlcv_provider.fetch_ohlcv.return_value = _candles(["2024-01-01"])
        return LiveBot(Strategy(), ohlcv_provider)

    return _create


def test_a_declared_leverage_is_set_before_the_strategy_books(
    create_margin_target_livebot,
):
    events: list[str] = []

    asyncio.run(create_margin_target_livebot(events, held_leverage=3.0).run())

    assert events == ["set_leverage", "book at 5.0"]


def test_a_leverage_moved_by_hand_is_set_back_before_the_strategy_books(
    create_margin_target_livebot, exchange
):
    events: list[str] = []

    asyncio.run(create_margin_target_livebot(events, held_leverage=2.0).run())

    exchange.set_leverage.assert_awaited_once_with(DOGE, 5.0)
    assert events[-1] == "book at 5.0"


def test_a_refused_leverage_is_warned_about_and_the_candle_still_trades(
    create_margin_target_livebot, exchange, caplog
):
    events: list[str] = []
    livebot = create_margin_target_livebot(events, held_leverage=3.0)
    exchange.set_leverage.side_effect = ExchangeRecoverableError("position open")

    with caplog.at_level(logging.WARNING):
        asyncio.run(livebot.run())

    assert "SetLeverageAction (DOGE/USDT:USDT)" in caplog.text
    assert events == ["book at 3.0"]
    exchange.place_market_order.assert_awaited_once()
