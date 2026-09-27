import asyncio
import sys
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import pytz

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from robottraderslab import Symbol
from robottraderslab._core import TimeFrame, to_seconds
from robottraderslab.ohlcv_provider.ccxt_ohlcv_adapter import EXCHANGES
from robottraderslab.ohlcv_provider.exchange.ohlcv_provider import (
    ExchangeOHLCVProvider,
)

pytestmark = pytest.mark.integration

# =============================================================================
# Configuration: Test Parameters
# =============================================================================


DEFAULT_EXCHANGE_NAME = "ccxt_bitget"
TEST_SYMBOL = "BTC/USDT:USDT"
TEST_TIMEFRAME = "1h"
INTEGRATION_BATCH_WINDOWS = [2]

# =============================================================================
# Fixtures: Test Infrastructure
# =============================================================================


@pytest.fixture(scope="module")
def batch_size() -> int:
    """Get the batch size for the default exchange."""
    exchange_name = DEFAULT_EXCHANGE_NAME.replace("ccxt_", "")
    return EXCHANGES[exchange_name]["limit_size_request"]


@pytest.fixture(scope="module")
def storage_dir() -> Iterator[str]:
    """Isolated temporary storage for each test module."""
    td = TemporaryDirectory()
    try:
        yield td.name
    finally:
        td.cleanup()


@pytest.fixture(scope="module")
def exchange_name() -> str:
    """Default exchange for testing."""
    return DEFAULT_EXCHANGE_NAME


@pytest.fixture(scope="module")
def provider_factory(
    storage_dir: str,
) -> Callable[[str, str | None, str | None], ExchangeOHLCVProvider]:
    """Factory to create providers with different date ranges."""

    def _factory(
        exchange: str = DEFAULT_EXCHANGE_NAME,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> ExchangeOHLCVProvider:
        provider = ExchangeOHLCVProvider(
            exchange=exchange,
            storage_dir=storage_dir,
        )
        if start_date and end_date:
            provider.set_dates(start_date, end_date)
        return provider

    return _factory


@pytest.fixture(scope="module")
def symbol_btc() -> Symbol:
    """Test symbol for BTC/USDT perpetual."""
    return Symbol.create(TEST_SYMBOL)


@pytest.fixture(scope="module")
def timeframe_1h() -> str:
    """1-hour timeframe for testing."""
    return TEST_TIMEFRAME


def download_ohlcv_multi_batch(
    num_batch_windows: int,
    provider_factory: Callable[[str, str | None, str | None], ExchangeOHLCVProvider],
    exchange_name: str,
    symbol: Symbol,
    timeframe: str,
    batch_size: int,
) -> tuple[pd.DataFrame, tuple[str, str, datetime, datetime]]:
    """
    Utility function: Download OHLCV data spanning multiple API batches.

    This is a pure utility that handles the download mechanics without any assertions.
    It creates a date range, downloads the data, and returns both the data and metadata.

    Args:
        num_batch_windows: Number of API batch windows to span
        provider_factory: Factory for creating OHLCV providers
        exchange_name: Exchange to download from
        symbol: Trading symbol
        timeframe: Timeframe string
        batch_size: Size of each API batch for the exchange

    Returns:
        Tuple of (downloaded_dataframe, (start_str, end_str, start_dt, end_dt))
    """
    start_str, end_str, start_dt, end_dt = _create_date_range(
        num_batch_windows, batch_size
    )

    provider = provider_factory(exchange_name, start_str, end_str)
    df = provider.fetch_ohlcv(symbol, timeframe)

    return df, (start_str, end_str, start_dt, end_dt)


def _create_date_range(
    num_batch_windows: int, batch_size: int
) -> tuple[str, str, datetime, datetime]:
    """
    Create a date range spanning multiple batches.

    Args:
        num_batch_windows: Number of complete batches plus partial last batch
        batch_size: Size of each API batch for the exchange

    Returns:
        Tuple of (start_str, end_str, start_dt, end_dt)
    """
    test_range_hours = (num_batch_windows * batch_size) + max(1, batch_size // 2)

    end_dt = datetime.now(pytz.UTC).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(hours=test_range_hours)

    return (
        start_dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
        end_dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
        start_dt,
        end_dt,
    )


# =============================================================================
# Gap Detection: Core Validation Logic
# =============================================================================


def assert_no_gaps(df: pd.DataFrame, timeframe: TimeFrame) -> None:
    """
    Comprehensive gap detection for OHLCV data.

    Validates:
    1. DataFrame is not empty
    2. Index is monotonically increasing (sorted)
    3. No duplicate timestamps
    4. All intervals between consecutive rows match expected timeframe
    5. Total span matches expected number of steps (catches missing blocks)

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        timeframe: Expected timeframe string (e.g., "1h")

    Raises:
        AssertionError: If any gap condition is detected
    """
    assert not df.empty, "DataFrame is empty"
    assert df.index.is_monotonic_increasing, "Index is not sorted ascending"
    assert df.index.is_unique, "Duplicate timestamps found"

    expected_delta = pd.Timedelta(seconds=to_seconds(timeframe))
    diffs = df.index.to_series().diff().dropna()
    bad = diffs[diffs != expected_delta]
    if not bad.empty:
        exp_m = int(expected_delta.total_seconds() // 60)
        parts = [
            f"{idx.isoformat()} obs={int(delta.total_seconds() // 60)}m exp={exp_m}m"
            for idx, delta in bad.items()
        ]
        raise AssertionError(
            "Inconsistent intervals between rows; all mismatches: " + " | ".join(parts)
        )

    first_ts = df.index[0]
    last_ts = df.index[-1]
    span = last_ts - first_ts
    expected_steps = int(span / expected_delta)
    if expected_steps != (len(df) - 1):
        exp_m = int(expected_delta.total_seconds() // 60)
        gaps: list[str] = []
        for ts, delta in diffs.items():
            if delta != expected_delta:
                expected_prev = ts - expected_delta
                obs_m = int(delta.total_seconds() // 60)
                gaps.append(
                    f"before={ts.isoformat()} prev~={expected_prev.isoformat()} obs={obs_m}m exp={exp_m}m"
                )
        raise AssertionError(
            "Expected {exp} steps, got {got}; gaps: ".format(
                exp=expected_steps, got=(len(df) - 1)
            )
            + " | ".join(gaps)
        )


# =============================================================================
# Validation Tests: Ensure Gap Detector Works
# =============================================================================


@pytest.fixture
def valid_sample_frame() -> pd.DataFrame:
    """Create a small valid OHLCV frame for testing gap detection."""
    dates = pd.date_range(
        "2024-01-01T00:00:00+00:00", periods=10, freq=TEST_TIMEFRAME, tz="UTC"
    )
    return pd.DataFrame(
        {
            "open": range(100, 110),
            "high": range(101, 111),
            "low": range(99, 109),
            "close": range(100, 110),
            "volume": range(1000, 1100, 10),
        },
        index=dates,
    )


@pytest.fixture
def gapped_sample_frame(valid_sample_frame: pd.DataFrame) -> pd.DataFrame:
    """Remove middle rows from valid frame to create gaps."""
    # Remove rows 3, 4, 5 to create a clear gap
    to_drop = valid_sample_frame.index[3:6]
    return valid_sample_frame.drop(index=to_drop)


def test_gap_detector_passes_on_valid_data(valid_sample_frame, timeframe_1h):
    """Verify gap detector passes on continuous data."""
    assert_no_gaps(valid_sample_frame, timeframe_1h)


def test_gap_detector_catches_missing_rows(gapped_sample_frame, timeframe_1h):
    """Verify gap detector fails on data with missing rows."""
    with pytest.raises(AssertionError):
        assert_no_gaps(gapped_sample_frame, timeframe_1h)


# =============================================================================
# Integration Tests: Real Download Scenarios
# =============================================================================


@pytest.mark.parametrize("num_batch_windows", INTEGRATION_BATCH_WINDOWS)
def test_download_then_file_load_both_gapless(
    num_batch_windows: int,
    provider_factory,
    exchange_name,
    symbol_btc,
    timeframe_1h,
    batch_size,
    monkeypatch,
):
    """
    Combined test: Download multi-batch data, then verify file-based loading is also gapless.

    This test efficiently combines both scenarios for each batch window count:
    1. Downloads data spanning multiple API batches and verifies no gaps
    2. Loads a subset from persisted files and verifies no gaps in loaded data
    3. Ensures the file-based loading doesn't trigger additional API calls

    This approach is more efficient than separate tests and ensures the same
    data is used for both download and file-load validation.
    """

    ### Step 1: Download OHLCV data spanning multiple API batches
    df_downloaded, (_, _, start_dt, end_dt) = download_ohlcv_multi_batch(
        num_batch_windows,
        provider_factory,
        exchange_name,
        symbol_btc,
        timeframe_1h,
        batch_size,
    )

    ### Step 2: Validate downloaded data has no gaps
    assert df_downloaded.index.min() >= start_dt, (
        "DOWNLOAD PHASE: Downloaded data starts before requested range"
    )
    assert df_downloaded.index.max() <= end_dt, (
        "DOWNLOAD PHASE: Downloaded data ends after requested range"
    )

    expected_min_candles = num_batch_windows * batch_size
    assert len(df_downloaded) >= expected_min_candles, (
        f"DOWNLOAD PHASE: Expected at least {expected_min_candles} candles for {num_batch_windows} batches"
    )

    try:
        assert_no_gaps(df_downloaded, timeframe_1h)
    except AssertionError as e:
        raise AssertionError(f"DOWNLOAD PHASE GAP DETECTED: {str(e)}") from e

    ### Step 3: Create subset window for file-based loading test
    total_hours = (end_dt - start_dt).total_seconds() / 3600
    subset_hours = min(100, max(10, int(total_hours / 4)))

    cache_start = start_dt + timedelta(hours=subset_hours)
    cache_end = end_dt - timedelta(hours=subset_hours)
    cache_start_str = cache_start.strftime("%Y-%m-%dT%H:%M:%S%z")
    cache_end_str = cache_end.strftime("%Y-%m-%dT%H:%M:%S%z")

    ### Step 4: Set up guard against exchange API calls during file loading
    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("Exchange API called during file-only operation")

    monkeypatch.setattr(
        "robottraderslab.ohlcv_provider.ccxt_ohlcv_adapter.CcxtOhlcvAdapter.fetch_ohlcv",
        _should_not_be_called,
        raising=True,
    )

    ### Step 5: Load subset from persisted files
    provider = provider_factory(exchange_name, cache_start_str, cache_end_str)
    df_from_files = provider.fetch_ohlcv(symbol_btc, timeframe_1h)

    ### Step 6: Validate file-loaded data has no gaps
    assert not df_from_files.empty, "FILE LOAD PHASE: File-loaded data is empty"
    assert df_from_files.index.min() >= cache_start, (
        "FILE LOAD PHASE: File-loaded data starts before requested range"
    )
    assert df_from_files.index.max() <= cache_end, (
        "FILE LOAD PHASE: File-loaded data ends after requested range"
    )

    try:
        assert_no_gaps(df_from_files, timeframe_1h)
    except AssertionError as e:
        raise AssertionError(f"FILE LOAD PHASE GAP DETECTED: {str(e)}") from e
