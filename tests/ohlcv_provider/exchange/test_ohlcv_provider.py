import asyncio
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import pytz

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from robottraderslab import Symbol
from robottraderslab._core import (
    OHLCVRequirements,
    TimeFrame,
    assemble_all_ohlcvs_from_requirements,
    assemble_available_ohlcvs_from_requirements,
    to_seconds,
)
from robottraderslab.exceptions import ExchangeCriticalError, StrategyCriticalError
from robottraderslab.exchanges import OhlcvAdapterProtocol, OhlcvData
from robottraderslab.ohlcv_provider import ExchangeOHLCVProvider
from robottraderslab.ohlcv_provider.ccxt_ohlcv_adapter import EXCHANGES
from robottraderslab.ohlcv_provider.exchange.ohlcv_cache import _OHLCV_CACHE
from robottraderslab.ohlcv_provider.repositories.null_ohlcv_repository import (
    NullOhlcvRepository,
)

_FRIDAY = 4
_SATURDAY = 5
_SUNDAY = 6
_WEEKLY_CLOSE_HOUR = 21
_WEEKLY_OPEN_HOUR = 21
_HOLIDAY_START = pd.Timestamp("2024-01-01", tz="UTC")
_HOLIDAY_END = pd.Timestamp("2024-01-03", tz="UTC")
_CLOSED_MONTH_START = pd.Timestamp("2024-06-01", tz="UTC")
_CLOSED_MONTH_END = pd.Timestamp("2024-07-01", tz="UTC")


def _weekend(index: pd.DatetimeIndex) -> np.ndarray:
    return (
        ((index.dayofweek == _FRIDAY) & (index.hour >= _WEEKLY_CLOSE_HOUR))
        | (index.dayofweek == _SATURDAY)
        | ((index.dayofweek == _SUNDAY) & (index.hour < _WEEKLY_OPEN_HOUR))
    )


def _settled_candle_stamps(
    start_date: datetime, end_date: datetime, timeframe: TimeFrame
) -> pd.DatetimeIndex:
    """Stamps of the timeframe grid inside an inclusive window."""
    step = pd.Timedelta(seconds=to_seconds(timeframe))
    return pd.date_range(
        start=pd.Timestamp(start_date).ceil(step), end=end_date, freq=step
    )


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}
    )


@pytest.fixture
def provider_mocks(sample_dataframe: pd.DataFrame):
    """Mock all external dependencies for the provider."""
    with (
        patch(
            "robottraderslab.ohlcv_provider.exchange.ohlcv_provider._create_repository"
        ) as mock_repo_factory,
        patch(
            "robottraderslab.ohlcv_provider.exchange.ohlcv_provider.fetch_ohlcv_with"
        ) as mock_fetch,
    ):
        mock_repo = Mock()
        mock_repo_factory.return_value = mock_repo

        async def fake_fetch(**kwargs):
            return sample_dataframe

        mock_fetch.side_effect = fake_fetch

        yield {
            "mock_repo_factory": mock_repo_factory,
            "mock_fetch": mock_fetch,
            "mock_repo": mock_repo,
        }


@pytest.fixture
def shared_provider(
    provider_mocks: dict[str, Mock], stub_exchange: str
) -> ExchangeOHLCVProvider:
    """Shared provider instance to avoid recreating in every test."""
    provider = ExchangeOHLCVProvider(
        exchange=stub_exchange,
        storage_dir="/tmp",
    )
    provider.set_dates("2024-01-01", "2024-01-02")
    return provider


@pytest.fixture
def date_conversion_provider(
    provider_mocks: dict[str, Mock], stub_exchange: str
) -> ExchangeOHLCVProvider:
    """Provider with different dates for date conversion testing."""
    provider = ExchangeOHLCVProvider(
        exchange=stub_exchange,
        storage_dir="storage-dir",
    )
    provider.set_dates("2024-01-01", "2025-01-01")
    return provider


@pytest.fixture
def path_provider(
    provider_mocks: dict[str, Mock], stub_exchange: str
) -> ExchangeOHLCVProvider:
    """Provider with Path storage_dir for path testing."""
    provider = ExchangeOHLCVProvider(
        exchange=stub_exchange,
        storage_dir=Path("/tmp/data"),
    )
    provider.set_dates("2024-01-01", "2024-01-02")
    return provider


class _FailingAdapter:
    """Adapter whose every candle request raises the failure it was created with."""

    def __init__(self, failure: Exception) -> None:
        self._failure = failure

    @classmethod
    def create(cls, *, failure: Exception, **_kwargs: object) -> "_FailingAdapter":
        return cls(failure)

    @classmethod
    def dataset_qualifiers(cls, **_kwargs: object) -> tuple[str, ...]:
        return ()

    async def __aenter__(self) -> "_FailingAdapter":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def fetch_ohlcv(self, *_args: object, **_kwargs: object) -> OhlcvData:
        raise self._failure

    @classmethod
    def market_open_mask(cls, index: pd.DatetimeIndex) -> None:
        return None


class _WeekendClosedAdapter(OhlcvAdapterProtocol):
    @classmethod
    def create(cls, **_kwargs: object) -> "_WeekendClosedAdapter":
        return cls()

    @classmethod
    def dataset_qualifiers(cls, **_kwargs: object) -> tuple[str, ...]:
        return ()

    async def __aenter__(self) -> "_WeekendClosedAdapter":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def fetch_ohlcv(self, *_args: object, **_kwargs: object) -> OhlcvData:
        raise NotImplementedError

    @classmethod
    def market_open_mask(cls, index: pd.DatetimeIndex) -> np.ndarray:
        return ~_weekend(index)


class _WeekendAndHolidayClosedAdapter(_WeekendClosedAdapter):
    @classmethod
    def market_open_mask(cls, index: pd.DatetimeIndex) -> np.ndarray:
        open_mask = super().market_open_mask(index)
        holiday = (index >= _HOLIDAY_START) & (index < _HOLIDAY_END)
        return open_mask & ~holiday


class _OneMonthClosedAdapter(OhlcvAdapterProtocol):
    @classmethod
    def create(cls, **_kwargs: object) -> "_OneMonthClosedAdapter":
        return cls()

    @classmethod
    def dataset_qualifiers(cls, **_kwargs: object) -> tuple[str, ...]:
        return ()

    async def __aenter__(self) -> "_OneMonthClosedAdapter":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def fetch_ohlcv(self, *_args: object, **_kwargs: object) -> OhlcvData:
        raise NotImplementedError

    @classmethod
    def market_open_mask(cls, index: pd.DatetimeIndex) -> np.ndarray:
        closed = (index >= _CLOSED_MONTH_START) & (index < _CLOSED_MONTH_END)
        return ~closed


class _AlwaysClosedAdapter(OhlcvAdapterProtocol):
    @classmethod
    def create(cls, **_kwargs: object) -> "_AlwaysClosedAdapter":
        return cls()

    @classmethod
    def dataset_qualifiers(cls, **_kwargs: object) -> tuple[str, ...]:
        return ()

    async def __aenter__(self) -> "_AlwaysClosedAdapter":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def fetch_ohlcv(self, *_args: object, **_kwargs: object) -> OhlcvData:
        raise NotImplementedError

    @classmethod
    def market_open_mask(cls, index: pd.DatetimeIndex) -> np.ndarray:
        return np.zeros(len(index), dtype=bool)


class _NeverBuiltAdapter(_WeekendClosedAdapter):
    @classmethod
    def create(cls, **_kwargs: object) -> "_NeverBuiltAdapter":
        raise AssertionError("built to read a calendar its class already answers")


class TestExchangeOHLCVProvider:
    """Tests for ExchangeOHLCVProvider class."""

    def test_implements_ohlcv_provider_protocol(
        self, shared_provider: ExchangeOHLCVProvider
    ):
        """Test that provider correctly implements the OHLCVProviderProtocol."""
        assert hasattr(shared_provider, "fetch_ohlcv")

    @pytest.mark.parametrize("storage_dir", ["/tmp", Path("/data"), "storage"])
    def test_accepts_various_parameter_types(
        self, provider_mocks, stub_exchange, storage_dir
    ):
        """Test provider constructor accepts different parameter types."""
        ExchangeOHLCVProvider(
            exchange=stub_exchange,
            storage_dir=storage_dir,
        )

    def test_flat_storage_dir_defaults_to_parquet(self, provider_mocks, stub_exchange):
        ExchangeOHLCVProvider(exchange=stub_exchange, storage_dir="/tmp")

        args, kwargs = provider_mocks["mock_repo_factory"].call_args
        assert args[0] == "parquet"
        assert kwargs["dir"] == "/tmp"

    def test_nested_storage_config_forwards_to_factory(
        self, provider_mocks, stub_exchange
    ):
        ExchangeOHLCVProvider(
            exchange=stub_exchange,
            storage={"type": "parquet", "dir": "/tmp", "compression": "snappy"},
        )

        args, kwargs = provider_mocks["mock_repo_factory"].call_args
        assert args[0] == "parquet"
        assert kwargs["dir"] == "/tmp"
        assert kwargs["compression"] == "snappy"

    def test_nested_storage_defaults_type_to_parquet(
        self, provider_mocks, stub_exchange
    ):
        ExchangeOHLCVProvider(exchange=stub_exchange, storage={"dir": "/tmp"})

        args, kwargs = provider_mocks["mock_repo_factory"].call_args
        assert args[0] == "parquet"
        assert kwargs["dir"] == "/tmp"

    def test_omitting_storage(self, stub_exchange):
        provider = ExchangeOHLCVProvider(exchange=stub_exchange)

        assert isinstance(provider._repository, NullOhlcvRepository)

    def test_a_qualified_dataset_takes_its_own_name(
        self, provider_mocks, stub_exchange
    ):
        provider = ExchangeOHLCVProvider(
            exchange=stub_exchange, storage_dir="/tmp", variant="demo"
        )

        assert provider._dataset_name == f"{stub_exchange}-demo"

    def test_a_source_on_its_defaults_keeps_its_own_name(
        self, provider_mocks, stub_exchange
    ):
        provider = ExchangeOHLCVProvider(exchange=stub_exchange, storage_dir="/tmp")

        assert provider._dataset_name == stub_exchange

    def test_a_qualified_dataset_stores_under_its_own_key(
        self, provider_mocks, stub_exchange
    ):
        ExchangeOHLCVProvider(
            exchange=stub_exchange, storage_dir="/tmp", variant="demo"
        )

        args, _ = provider_mocks["mock_repo_factory"].call_args
        assert args[1] == f"{stub_exchange}-demo"

    def test_unsupported_storage_type(self, stub_exchange):
        from robottraderslab.ohlcv_provider.exceptions import StorageTypeError

        with pytest.raises(StorageTypeError, match="Unsupported storage_type: sqlite"):
            ExchangeOHLCVProvider(
                exchange=stub_exchange, storage={"type": "sqlite", "dir": "/tmp"}
            )

    def test_fetch_ohlcv_returns_dataframe(
        self, shared_provider: ExchangeOHLCVProvider
    ):
        """Test that fetch_ohlcv returns a DataFrame."""
        result = shared_provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1h")
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_wires_dependencies_correctly(
        self, date_conversion_provider: ExchangeOHLCVProvider, provider_mocks
    ):
        """Test that provider correctly wires its dependencies."""
        date_conversion_provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "4h")

        mocks = provider_mocks
        mocks["mock_repo_factory"].assert_called_once()
        mocks["mock_fetch"].assert_called()

    def test_converts_storage_dir_to_path(
        self, path_provider: ExchangeOHLCVProvider, provider_mocks
    ):
        """Test that string storage_dir gets converted to Path object."""
        path_provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1h")

        _, kwargs = provider_mocks["mock_repo_factory"].call_args
        assert isinstance(kwargs["dir"], Path)

    def test_converts_string_dates_to_datetime(
        self, date_conversion_provider: ExchangeOHLCVProvider, provider_mocks
    ):
        _OHLCV_CACHE.clear()

        async def fake_fetch_with(**kwargs):
            assert isinstance(kwargs["start_date"], datetime)
            assert isinstance(kwargs["end_date"], datetime)
            assert kwargs["start_date"] == datetime(2024, 1, 1, tzinfo=pytz.UTC)
            assert kwargs["end_date"] == datetime(2025, 1, 1, tzinfo=pytz.UTC)
            return pd.DataFrame()

        provider_mocks["mock_fetch"].side_effect = fake_fetch_with

        date_conversion_provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "4h")
        provider_mocks["mock_fetch"].assert_called_once()

    def test_uses_cache_on_repeated_calls(
        self, shared_provider: ExchangeOHLCVProvider, provider_mocks
    ):
        _OHLCV_CACHE.clear()

        symbol = Symbol.create("BTC/USDT:USDT")
        timeframe = "1h"

        result1 = shared_provider.fetch_ohlcv(symbol, timeframe)
        assert provider_mocks["mock_fetch"].call_count == 1

        result2 = shared_provider.fetch_ohlcv(symbol, timeframe)
        assert provider_mocks["mock_fetch"].call_count == 1

        pd.testing.assert_frame_equal(result1, result2)

    async def test_integration_with_assemble_all_ohlcvs_from_requirements(
        self, shared_provider: ExchangeOHLCVProvider
    ):
        requirements = OHLCVRequirements()
        requirements.add(Symbol.create("BTC/USDT:USDT"), "1h")

        assembled = await assemble_all_ohlcvs_from_requirements(
            shared_provider, requirements
        )

        assert assembled is not None

    @pytest.mark.parametrize(
        "failure",
        [
            ExchangeCriticalError("key revoked"),
            StrategyCriticalError("SOL/USD:USD is not a Kraken perpetual"),
        ],
        ids=["exchange_critical", "strategy_critical"],
    )
    async def test_a_critical_the_adapter_raises_stops_the_live_assembly(
        self, make_registered_adapter, failure
    ):
        make_registered_adapter(_FailingAdapter, "failing_exchange")
        _OHLCV_CACHE.clear()
        provider = ExchangeOHLCVProvider(exchange="failing_exchange", failure=failure)
        provider.set_dates("2025-01-01", "2025-01-03")
        requirements = OHLCVRequirements()
        requirements.add(Symbol.create("BTC/USDT:USDT"), "1h")

        with pytest.raises(type(failure), match=str(failure)):
            await assemble_available_ohlcvs_from_requirements(provider, requirements)

    async def test_fetch_ohlcvs_returns_frame_per_pair(
        self, shared_provider: ExchangeOHLCVProvider, provider_mocks
    ):
        _OHLCV_CACHE.clear()
        btc = Symbol.create("BTC/USDT:USDT")
        eth = Symbol.create("ETH/USDT:USDT")

        frames = await shared_provider.fetch_ohlcvs([(btc, "1h"), (eth, "4h")])

        assert set(frames.keys()) == {(btc, "1h"), (eth, "4h")}
        assert all(isinstance(frame, pd.DataFrame) for frame in frames.values())
        assert provider_mocks["mock_fetch"].call_count == 2

    async def test_fetch_ohlcvs_serves_cached_pairs_without_refetch(
        self, shared_provider: ExchangeOHLCVProvider, provider_mocks
    ):
        _OHLCV_CACHE.clear()
        btc = Symbol.create("BTC/USDT:USDT")
        cached_frame = (await shared_provider.fetch_ohlcvs([(btc, "1h")]))[(btc, "1h")]
        assert provider_mocks["mock_fetch"].call_count == 1

        frames = await shared_provider.fetch_ohlcvs([(btc, "1h")])

        assert provider_mocks["mock_fetch"].call_count == 1
        pd.testing.assert_frame_equal(frames[(btc, "1h")], cached_frame)

    async def test_fetch_ohlcvs_loads_duplicate_pairs_once(
        self, shared_provider: ExchangeOHLCVProvider, provider_mocks
    ):
        _OHLCV_CACHE.clear()
        btc = Symbol.create("BTC/USDT:USDT")

        frames = await shared_provider.fetch_ohlcvs([(btc, "1h"), (btc, "1h")])

        assert list(frames.keys()) == [(btc, "1h")]
        assert provider_mocks["mock_fetch"].call_count == 1

    async def test_fetch_ohlcvs_failure_cancels_remaining_loads(
        self, shared_provider: ExchangeOHLCVProvider, provider_mocks
    ):
        _OHLCV_CACHE.clear()
        btc = Symbol.create("BTC/USDT:USDT")
        eth = Symbol.create("ETH/USDT:USDT")
        blocked_cancelled = asyncio.Event()

        async def fetch_by_symbol(**kwargs):
            if kwargs["symbol"] == btc:
                raise ValueError("boom")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                blocked_cancelled.set()
                raise
            return pd.DataFrame()

        provider_mocks["mock_fetch"].side_effect = fetch_by_symbol

        with pytest.raises(ValueError, match="boom"):
            await shared_provider.fetch_ohlcvs([(btc, "1h"), (eth, "1h")])

        assert blocked_cancelled.is_set()

    def test_per_timeframe_lookback_in_live_mode(self, provider_mocks, stub_exchange):
        _OHLCV_CACHE.clear()

        provider = ExchangeOHLCVProvider(
            exchange=stub_exchange,
            storage_dir="/tmp",
        )

        lookbacks = {"1h": 100, "1d": 30}
        provider.set_required_lookbacks(lookbacks)

        fetch_calls = []

        async def capture_date_range(**kwargs):
            fetch_calls.append(
                {
                    "timeframe": kwargs["timeframe"],
                    "start_date": kwargs["start_date"],
                    "end_date": kwargs["end_date"],
                }
            )
            return pd.DataFrame()

        provider_mocks["mock_fetch"].side_effect = capture_date_range

        provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1h")
        provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1d")

        assert len(fetch_calls) == 2

        call_1h = next(c for c in fetch_calls if c["timeframe"] == "1h")
        call_1d = next(c for c in fetch_calls if c["timeframe"] == "1d")

        hourly_stamps = _settled_candle_stamps(
            call_1h["start_date"], call_1h["end_date"], "1h"
        )
        daily_stamps = _settled_candle_stamps(
            call_1d["start_date"], call_1d["end_date"], "1d"
        )

        assert len(hourly_stamps) == 101
        assert len(daily_stamps) == 31

    def test_shared_reference_time_across_symbols_in_live_mode(
        self, provider_mocks, stub_exchange
    ):
        _OHLCV_CACHE.clear()

        provider = ExchangeOHLCVProvider(
            exchange=stub_exchange,
            storage_dir="/tmp",
        )
        provider.set_required_lookbacks({"1h": 100})

        fetch_calls: list[dict] = []

        async def capture_date_range(**kwargs):
            fetch_calls.append(
                {
                    "symbol": kwargs["symbol"],
                    "end_date": kwargs["end_date"],
                }
            )
            return pd.DataFrame()

        provider_mocks["mock_fetch"].side_effect = capture_date_range

        provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1h")
        provider.fetch_ohlcv(Symbol.create("ETH/USDT:USDT"), "1h")
        provider.fetch_ohlcv(Symbol.create("SOL/USDT:USDT"), "1h")

        assert len(fetch_calls) == 3
        assert fetch_calls[0]["end_date"] == fetch_calls[1]["end_date"]
        assert fetch_calls[1]["end_date"] == fetch_calls[2]["end_date"]

    def test_lookback_extends_start_date_in_backtest_mode(
        self, provider_mocks, stub_exchange
    ):
        _OHLCV_CACHE.clear()

        provider = ExchangeOHLCVProvider(
            exchange=stub_exchange,
            storage_dir="/tmp",
        )
        provider.set_dates("2024-01-01", "2024-12-31")

        provider.set_required_lookbacks({"1h": 100})

        fetch_calls = []

        async def capture_date_range(**kwargs):
            fetch_calls.append(
                {
                    "start_date": kwargs["start_date"],
                    "end_date": kwargs["end_date"],
                }
            )
            return pd.DataFrame()

        provider_mocks["mock_fetch"].side_effect = capture_date_range

        provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1h")

        assert len(fetch_calls) == 1
        expected_start = datetime(2024, 1, 1, tzinfo=pytz.UTC) - pd.Timedelta(hours=100)
        assert fetch_calls[0]["start_date"] == expected_start
        assert fetch_calls[0]["end_date"] == datetime(2024, 12, 31, tzinfo=pytz.UTC)

    def test_no_lookback_uses_exact_dates_in_backtest_mode(
        self, provider_mocks, stub_exchange
    ):
        _OHLCV_CACHE.clear()

        provider = ExchangeOHLCVProvider(
            exchange=stub_exchange,
            storage_dir="/tmp",
        )
        provider.set_dates("2024-01-01", "2024-12-31")

        fetch_calls = []

        async def capture_date_range(**kwargs):
            fetch_calls.append(
                {
                    "start_date": kwargs["start_date"],
                    "end_date": kwargs["end_date"],
                }
            )
            return pd.DataFrame()

        provider_mocks["mock_fetch"].side_effect = capture_date_range

        provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1h")

        assert len(fetch_calls) == 1
        assert fetch_calls[0]["start_date"] == datetime(2024, 1, 1, tzinfo=pytz.UTC)
        assert fetch_calls[0]["end_date"] == datetime(2024, 12, 31, tzinfo=pytz.UTC)

    @pytest.mark.parametrize(
        ("timeframe", "lookback", "candle_seconds"),
        [
            ("1h", 24, 3600),
            ("1d", 7, 24 * 3600),
            ("5m", 12, 300),
            ("15m", 96, 900),
        ],
    )
    def test_live_range_ends_on_a_settled_candle_stamp(
        self, provider_mocks, timeframe, lookback, candle_seconds, stub_exchange
    ):
        _OHLCV_CACHE.clear()

        provider = ExchangeOHLCVProvider(
            exchange=stub_exchange,
            storage_dir="/tmp",
        )

        provider.set_required_lookbacks({timeframe: lookback})

        async def check_end_date(**kwargs):
            end = kwargs["end_date"]
            assert isinstance(end, datetime)
            assert int(end.timestamp()) % candle_seconds == 0
            return pd.DataFrame()

        provider_mocks["mock_fetch"].side_effect = check_end_date

        provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), timeframe)
        provider_mocks["mock_fetch"].assert_called_once()

    def test_repeated_live_cycles_inside_one_candle(
        self, provider_mocks, stub_exchange
    ):
        _OHLCV_CACHE.clear()
        cycles = 3
        provider = ExchangeOHLCVProvider(exchange=stub_exchange, storage_dir="/tmp")
        requested_windows: list[tuple[datetime, datetime]] = []

        async def capture_window(**kwargs):
            requested_windows.append((kwargs["start_date"], kwargs["end_date"]))
            return pd.DataFrame()

        provider_mocks["mock_fetch"].side_effect = capture_window

        for _ in range(cycles):
            provider.set_required_lookbacks({"1h": 100})
            provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1h")

        assert len(requested_windows) == 1
        asked_start, asked_end = requested_windows[0]
        assert (asked_end.minute, asked_end.second, asked_end.microsecond) == (0, 0, 0)
        assert (asked_start.minute, asked_start.second, asked_start.microsecond) == (
            0,
            0,
            0,
        )

    @pytest.mark.parametrize(
        ("timeframe", "lookback"),
        [
            ("1h", 24),
            ("1d", 7),
            ("5m", 12),
            ("15m", 96),
        ],
    )
    def test_live_delivers_the_lookback_before_the_candle_it_books(
        self, provider_mocks, timeframe, lookback, stub_exchange
    ):
        _OHLCV_CACHE.clear()

        provider = ExchangeOHLCVProvider(exchange=stub_exchange, storage_dir="/tmp")
        provider.set_required_lookbacks({timeframe: lookback})

        async def answer_with_the_settled_candles(**kwargs):
            stamps = _settled_candle_stamps(
                kwargs["start_date"], kwargs["end_date"], timeframe
            )
            return pd.DataFrame(
                {
                    column: range(len(stamps))
                    for column in ("open", "high", "low", "close", "volume")
                },
                index=stamps,
            )

        provider_mocks["mock_fetch"].side_effect = answer_with_the_settled_candles

        settled_candles = provider.fetch_ohlcv(
            Symbol.create("BTC/USDT:USDT"), timeframe
        )

        assert len(settled_candles) == lookback + 1

    @pytest.mark.parametrize(
        "reference_time",
        [
            datetime(2024, 1, 10, 12, tzinfo=pytz.UTC),
            datetime(2024, 1, 14, 12, tzinfo=pytz.UTC),
        ],
        ids=["newest settled stamp open", "newest settled stamp closed"],
    )
    def test_live_lookback_counts_only_open_stamps_on_a_closed_calendar(
        self, provider_mocks, make_registered_adapter, reference_time
    ):
        _OHLCV_CACHE.clear()
        make_registered_adapter(_WeekendClosedAdapter, "weekend_closed")

        provider = ExchangeOHLCVProvider(exchange="weekend_closed", storage_dir="/tmp")
        lookback = 10
        provider.set_required_lookbacks({"1d": lookback})
        provider._lookback_reference_time = reference_time

        fetch_calls: list[tuple[datetime, datetime]] = []

        async def capture_date_range(**kwargs):
            fetch_calls.append((kwargs["start_date"], kwargs["end_date"]))
            return pd.DataFrame()

        provider_mocks["mock_fetch"].side_effect = capture_date_range

        provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1d")

        start_date, end_date = fetch_calls[0]
        served_stamps = pd.date_range(start_date, end_date, freq="1D")
        open_stamps = served_stamps[~_weekend(served_stamps)]
        assert len(open_stamps) == lookback + 1

    def test_backtest_lookback_counts_only_open_stamps_on_a_closed_calendar(
        self, provider_mocks, make_registered_adapter
    ):
        _OHLCV_CACHE.clear()
        make_registered_adapter(_WeekendClosedAdapter, "weekend_closed")

        provider = ExchangeOHLCVProvider(exchange="weekend_closed", storage_dir="/tmp")
        trading_start = datetime(2024, 1, 15, tzinfo=pytz.UTC)  # a Monday
        provider.set_dates("2024-01-15", "2024-02-01")
        lookback = 10
        provider.set_required_lookbacks({"1d": lookback})

        fetch_calls: list[datetime] = []

        async def capture_start(**kwargs):
            fetch_calls.append(kwargs["start_date"])
            return pd.DataFrame()

        provider_mocks["mock_fetch"].side_effect = capture_start

        provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1d")

        returned_start = fetch_calls[0]
        stamps_before_trading = pd.date_range(returned_start, trading_start, freq="1D")[
            :-1
        ]
        open_before_trading = stamps_before_trading[~_weekend(stamps_before_trading)]
        assert len(open_before_trading) == lookback

    def test_a_start_off_the_candle_grid_warms_up_the_candle_after_it(
        self, provider_mocks, make_registered_adapter
    ):
        _OHLCV_CACHE.clear()
        make_registered_adapter(_WeekendClosedAdapter, "weekend_closed")

        provider = ExchangeOHLCVProvider(exchange="weekend_closed", storage_dir="/tmp")
        provider.set_dates("2024-01-15 22:37:21", "2024-02-01")
        lookback = 10
        provider.set_required_lookbacks({"1d": lookback})

        fetch_calls: list[datetime] = []

        async def capture_start(**kwargs):
            fetch_calls.append(kwargs["start_date"])
            return pd.DataFrame()

        provider_mocks["mock_fetch"].side_effect = capture_start

        provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1d")

        returned_start = fetch_calls[0]
        first_traded = datetime(2024, 1, 16, tzinfo=pytz.UTC)
        served_before_trading = pd.date_range(
            returned_start, first_traded, freq="1D", inclusive="left"
        )
        assert returned_start == returned_start.replace(hour=0, minute=0, second=0)
        assert len(served_before_trading[~_weekend(served_before_trading)]) == lookback

    def test_a_monthly_lookback_skips_a_closed_month(
        self, provider_mocks, make_registered_adapter
    ):
        _OHLCV_CACHE.clear()
        make_registered_adapter(_OneMonthClosedAdapter, "one_month_closed")

        provider = ExchangeOHLCVProvider(
            exchange="one_month_closed", storage_dir="/tmp"
        )
        trading_start = datetime(2024, 9, 1, tzinfo=pytz.UTC)
        provider.set_dates("2024-09-01", "2024-10-01")
        lookback = 3
        provider.set_required_lookbacks({"1M": lookback})

        fetch_calls: list[datetime] = []

        async def capture_start(**kwargs):
            fetch_calls.append(kwargs["start_date"])
            return pd.DataFrame()

        provider_mocks["mock_fetch"].side_effect = capture_start

        provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1M")

        returned_start = fetch_calls[0]
        month_stamps = pd.date_range(
            returned_start, trading_start, freq=pd.offsets.MonthBegin(1)
        )[:-1]
        closed = (month_stamps >= _CLOSED_MONTH_START) & (
            month_stamps < _CLOSED_MONTH_END
        )
        assert closed.sum() == 1
        assert (~closed).sum() == lookback

    def test_a_holiday_widens_the_window_past_the_weekend_alone(
        self, provider_mocks, make_registered_adapter
    ):
        _OHLCV_CACHE.clear()
        lookback = 10
        fetch_calls: list[datetime] = []

        async def capture_start(**kwargs):
            fetch_calls.append(kwargs["start_date"])
            return pd.DataFrame()

        provider_mocks["mock_fetch"].side_effect = capture_start

        make_registered_adapter(_WeekendClosedAdapter, "weekend_only")
        weekend_only_provider = ExchangeOHLCVProvider(
            exchange="weekend_only", storage_dir="/tmp"
        )
        weekend_only_provider.set_dates("2024-01-15", "2024-02-01")
        weekend_only_provider.set_required_lookbacks({"1d": lookback})
        weekend_only_provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1d")
        weekend_only_start = fetch_calls[-1]

        make_registered_adapter(_WeekendAndHolidayClosedAdapter, "weekend_and_holiday")
        with_holiday_provider = ExchangeOHLCVProvider(
            exchange="weekend_and_holiday", storage_dir="/tmp"
        )
        with_holiday_provider.set_dates("2024-01-15", "2024-02-01")
        with_holiday_provider.set_required_lookbacks({"1d": lookback})
        with_holiday_provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1d")
        with_holiday_start = fetch_calls[-1]

        assert with_holiday_start < weekend_only_start

    def test_a_calendar_closed_for_the_whole_span_stops_the_run(
        self, provider_mocks, make_registered_adapter
    ):
        _OHLCV_CACHE.clear()
        make_registered_adapter(_AlwaysClosedAdapter, "always_closed")

        provider = ExchangeOHLCVProvider(exchange="always_closed", storage_dir="/tmp")
        provider.set_required_lookbacks({"1d": 10})

        with pytest.raises(StrategyCriticalError) as failure:
            provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1d")

        assert "BTC/USDT:USDT" in str(failure.value)
        assert "1d" in str(failure.value)

    def test_reading_the_calendar_builds_no_adapter(
        self, provider_mocks, make_registered_adapter
    ):
        _OHLCV_CACHE.clear()
        make_registered_adapter(_NeverBuiltAdapter, "never_built")

        provider = ExchangeOHLCVProvider(exchange="never_built", storage_dir="/tmp")
        provider.set_required_lookbacks({"1d": 10})

        async def fake_fetch(**kwargs):
            return pd.DataFrame()

        provider_mocks["mock_fetch"].side_effect = fake_fetch

        provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1d")

    def test_get_all_cached_ohlcv_for_symbol(self, shared_provider):
        _OHLCV_CACHE.clear()
        btc = Symbol.create("BTC/USDT:USDT")

        shared_provider.fetch_ohlcv(btc, "1h")

        cached_ohlcv = shared_provider.get_all_cached_ohlcv_for_symbol(btc)

        assert not cached_ohlcv.empty

    def test_parquet_storage_type(self, tmp_path, stub_exchange):
        ExchangeOHLCVProvider(
            exchange=stub_exchange,
            storage={"type": "parquet", "dir": str(tmp_path)},
        )

    def test_keeps_every_candle_the_source_returns(self, provider_mocks, stub_exchange):
        _OHLCV_CACHE.clear()

        provider = ExchangeOHLCVProvider(exchange=stub_exchange, storage_dir="/tmp")
        provider.set_dates("2024-01-01", "2024-01-02")

        three_row_df = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.0, 2.0, 3.0],
                "low": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "volume": [1.0, 2.0, 3.0],
            }
        )

        async def fake_fetch(**kwargs):
            return three_row_df

        provider_mocks["mock_fetch"].side_effect = fake_fetch

        candles = provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1h")

        assert list(candles["close"]) == [1.0, 2.0, 3.0]

    def test_a_12_hour_lookback_is_refused(self, provider_mocks, stub_exchange):
        from robottraderslab._core import OhlcvValidationError

        _OHLCV_CACHE.clear()
        provider = ExchangeOHLCVProvider(exchange=stub_exchange, storage_dir="/tmp")

        with pytest.raises(OhlcvValidationError, match="Unsupported timeframe"):
            provider.set_required_lookbacks({"12h": 10})

    def test_a_45_minute_lookback_is_supported(
        self, provider_mocks, sample_dataframe, stub_exchange
    ):
        _OHLCV_CACHE.clear()
        provider = ExchangeOHLCVProvider(exchange=stub_exchange, storage_dir="/tmp")
        provider.set_required_lookbacks({"45m": 10})

        candles = provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "45m")

        pd.testing.assert_frame_equal(candles, sample_dataframe)


class TestALiveWindowThroughCcxt:
    def test_a_lookback_wider_than_a_page_is_served_in_full(
        self, bitget_answering_from_its_recent_endpoint
    ):
        _OHLCV_CACHE.clear()
        page_candles = EXCHANGES["bitget"]["limit_size_request"]
        lookback = page_candles + page_candles // 4
        provider = ExchangeOHLCVProvider(exchange="ccxt_bitget")
        provider.set_required_lookbacks({"1h": lookback})

        served = provider.fetch_ohlcv(Symbol.create("BTC/USDT:USDT"), "1h")

        assert len(served) == lookback + 1
        assert not served.isna().any().any()
