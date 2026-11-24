import asyncio
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import pytz

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from robottraderslab import TimeFrame
from robottraderslab.ohlcv_provider.date_utils import is_timezone_aware
from robottraderslab.ohlcv_provider.exchange.gap_analysis import (
    compute_storage_gaps,
)


def _every_stamp_open(index: pd.DatetimeIndex) -> None:
    return None


def _make_datetime(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    """Create datetime with timezone awareness based on configuration."""
    dt = datetime(year, month, day, hour, minute, second)
    return dt.replace(tzinfo=pytz.UTC) if is_timezone_aware() else dt


def _make_dataframe_index(dates: list[datetime]) -> pd.DatetimeIndex:
    """Create DatetimeIndex with proper timezone handling for tests."""
    if is_timezone_aware():
        tz_aware_dates = [
            dt.replace(tzinfo=pytz.UTC) if dt.tzinfo is None else dt for dt in dates
        ]
        return pd.DatetimeIndex(tz_aware_dates, tz=pytz.UTC)
    else:
        naive_dates = [
            dt.replace(tzinfo=None) if dt.tzinfo is not None else dt for dt in dates
        ]
        return pd.DatetimeIndex(naive_dates)


@pytest.fixture
def start_date() -> datetime:
    """Standard start date for testing."""
    return _make_datetime(2024, 1, 1, 0, 0, 0)


@pytest.fixture
def end_date() -> datetime:
    """Standard end date for testing."""
    return _make_datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def timeframe() -> TimeFrame:
    """Standard timeframe for testing."""
    return "1h"


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Empty DataFrame with datetime index."""
    return pd.DataFrame(index=pd.DatetimeIndex([]))


@pytest.fixture
def single_point_dataframe() -> pd.DataFrame:
    """DataFrame with single data point."""
    return pd.DataFrame(
        {"close": [100.0]},
        index=_make_dataframe_index([_make_datetime(2024, 1, 1, 6, 0, 0)]),
    )


@pytest.fixture
def continuous_dataframe() -> pd.DataFrame:
    """DataFrame with continuous hourly data covering the full range."""
    dates = [_make_datetime(2024, 1, 1, h, 0, 0) for h in range(13)]
    return pd.DataFrame(
        {"close": range(len(dates))}, index=_make_dataframe_index(dates)
    )


@pytest.fixture
def dataframe_with_internal_gap() -> pd.DataFrame:
    """DataFrame with gap in the middle."""
    dates = [
        _make_datetime(2024, 1, 1, 0, 0, 0),
        _make_datetime(2024, 1, 1, 1, 0, 0),
        _make_datetime(2024, 1, 1, 2, 0, 0),
        _make_datetime(2024, 1, 1, 6, 0, 0),
        _make_datetime(2024, 1, 1, 7, 0, 0),
    ]
    return pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=_make_dataframe_index(dates))


@pytest.fixture
def dataframe_with_multiple_gaps() -> pd.DataFrame:
    """DataFrame with multiple internal gaps."""
    dates = [
        _make_datetime(2024, 1, 1, 0, 0, 0),
        _make_datetime(2024, 1, 1, 1, 0, 0),
        _make_datetime(2024, 1, 1, 3, 0, 0),
        _make_datetime(2024, 1, 1, 6, 0, 0),
        _make_datetime(2024, 1, 1, 7, 0, 0),
        _make_datetime(2024, 1, 1, 10, 0, 0),
    ]
    return pd.DataFrame(
        {"close": [1, 2, 3, 4, 5, 6]}, index=_make_dataframe_index(dates)
    )


@pytest.fixture
def dataframe_starts_after_window() -> pd.DataFrame:
    """DataFrame that starts after the requested window start."""
    dates = [_make_datetime(2024, 1, 1, h, 0, 0) for h in range(3, 10)]  # 3-9 hours
    return pd.DataFrame(
        {"close": range(len(dates))}, index=_make_dataframe_index(dates)
    )


@pytest.fixture
def dataframe_ends_before_window() -> pd.DataFrame:
    """DataFrame that ends before the requested window end."""
    dates = [_make_datetime(2024, 1, 1, h, 0, 0) for h in range(2, 9)]  # 2-8 hours
    return pd.DataFrame(
        {"close": range(len(dates))}, index=_make_dataframe_index(dates)
    )


@pytest.fixture
def dataframe_outside_window() -> pd.DataFrame:
    """DataFrame completely outside the requested window."""
    dates = [
        _make_datetime(2023, 12, 31, h, 0, 0) for h in range(20, 24)
    ]  # 20-23 hours
    return pd.DataFrame(
        {"close": range(len(dates))}, index=_make_dataframe_index(dates)
    )


@pytest.fixture
def dataframe_unsorted() -> pd.DataFrame:
    """DataFrame with unsorted timestamps."""
    dates = [
        _make_datetime(2024, 1, 1, 6, 0, 0),
        _make_datetime(2024, 1, 1, 2, 0, 0),
        _make_datetime(2024, 1, 1, 8, 0, 0),
        _make_datetime(2024, 1, 1, 4, 0, 0),
    ]
    return pd.DataFrame({"close": [6, 2, 8, 4]}, index=pd.DatetimeIndex(dates))


class TestComputeStorageGapsEmptyData:
    """Test cases for empty DataFrame scenarios."""

    def test_returns_full_range_when_dataframe_is_empty(
        self, empty_dataframe, start_date, end_date, timeframe
    ):
        gaps = compute_storage_gaps(
            empty_dataframe,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert gaps == [(start_date, end_date)]

    def test_returns_full_range_when_no_data_in_window(
        self, dataframe_outside_window, start_date, end_date, timeframe
    ):
        gaps = compute_storage_gaps(
            dataframe_outside_window,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert gaps == [(start_date, end_date)]


class TestComputeStorageGapsNoGaps:
    """Test cases for complete data coverage scenarios."""

    def test_returns_empty_when_continuous_data_covers_full_range(
        self, continuous_dataframe, start_date, end_date, timeframe
    ):
        gaps = compute_storage_gaps(
            continuous_dataframe,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert gaps == []

    def test_returns_empty_when_data_extends_beyond_window(self):
        # Create dates from 2023-12-31 22:00 to 2024-01-01 14:00 (17 hours total)
        dates = []
        # Add 2023-12-31 22:00 and 23:00
        dates.extend([_make_datetime(2023, 12, 31, h, 0, 0) for h in [22, 23]])
        # Add 2024-01-01 00:00 to 14:00
        dates.extend([_make_datetime(2024, 1, 1, h, 0, 0) for h in range(15)])

        dataframe = pd.DataFrame(
            {"close": range(len(dates))}, index=_make_dataframe_index(dates)
        )
        start_date = _make_datetime(2024, 1, 1, 0, 0, 0)
        end_date = _make_datetime(2024, 1, 1, 12, 0, 0)

        gaps = compute_storage_gaps(
            dataframe, start_date, end_date, "1h", market_open_mask=_every_stamp_open
        )

        assert gaps == []


class TestComputeStorageGapsMissingBefore:
    """Test cases for gaps before the first data point."""

    def test_returns_gap_before_first_when_data_starts_after_window(
        self, dataframe_starts_after_window, start_date, end_date, timeframe
    ):
        gaps = compute_storage_gaps(
            dataframe_starts_after_window,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert len(gaps) == 2
        assert gaps[0] == (start_date, _make_datetime(2024, 1, 1, 3, 0, 0))
        assert gaps[1] == (_make_datetime(2024, 1, 1, 9, 0, 0), end_date)

    def test_returns_gap_before_single_point_in_middle(
        self, single_point_dataframe, start_date, end_date, timeframe
    ):
        gaps = compute_storage_gaps(
            single_point_dataframe,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert len(gaps) == 2
        assert gaps[0] == (start_date, _make_datetime(2024, 1, 1, 6, 0, 0))
        assert gaps[1] == (_make_datetime(2024, 1, 1, 6, 0, 0), end_date)


class TestComputeStorageGapsMissingAfter:
    """Test cases for gaps after the last data point."""

    def test_returns_gap_after_last_when_data_ends_before_window(
        self, dataframe_ends_before_window, start_date, end_date, timeframe
    ):
        gaps = compute_storage_gaps(
            dataframe_ends_before_window,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert len(gaps) == 2
        assert gaps[0] == (start_date, _make_datetime(2024, 1, 1, 2, 0, 0))
        assert gaps[1] == (_make_datetime(2024, 1, 1, 8, 0, 0), end_date)


class TestComputeStorageGapsHead:
    """Test cases for the range before the first stored candle."""

    def test_start_inside_the_first_stored_candle(
        self, continuous_dataframe, end_date, timeframe
    ):
        start_date = _make_datetime(2024, 1, 1, 0, 30, 0)

        gaps = compute_storage_gaps(
            continuous_dataframe,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert gaps == []

    def test_start_exactly_one_candle_before_the_first_stored_one(
        self, continuous_dataframe, end_date, timeframe
    ):
        start_date = _make_datetime(2023, 12, 31, 23, 0, 0)

        gaps = compute_storage_gaps(
            continuous_dataframe,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert gaps == [(start_date, _make_datetime(2024, 1, 1, 0, 0, 0))]


class TestComputeStorageGapsTail:
    """Test cases for the range after the last stored candle."""

    def test_end_before_the_candle_after_the_last_stored_one_starts(
        self, continuous_dataframe, start_date, timeframe
    ):
        end_date = _make_datetime(2024, 1, 1, 12, 30, 0)

        gaps = compute_storage_gaps(
            continuous_dataframe,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert gaps == []

    def test_end_on_the_stamp_of_the_candle_after_the_last_stored_one(
        self, continuous_dataframe, start_date, timeframe
    ):
        last_stored = _make_datetime(2024, 1, 1, 12, 0, 0)
        end_date = _make_datetime(2024, 1, 1, 13, 0, 0)

        gaps = compute_storage_gaps(
            continuous_dataframe,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert gaps == [(last_stored, end_date)]

    def test_a_market_that_closed_and_reported_no_data(
        self, continuous_dataframe, start_date, timeframe
    ):
        last_stored = _make_datetime(2024, 1, 1, 12, 0, 0)
        end_date = _make_datetime(2024, 1, 2, 0, 0, 0)

        gaps = compute_storage_gaps(
            continuous_dataframe,
            start_date,
            end_date,
            timeframe,
            known_empty=[(last_stored, end_date)],
            market_open_mask=_every_stamp_open,
        )

        assert gaps == []


class TestComputeStorageGapsInternalGaps:
    """Test cases for gaps between data points."""

    def test_returns_internal_gap_when_single_gap_exists(
        self, dataframe_with_internal_gap, start_date, end_date, timeframe
    ):
        gaps = compute_storage_gaps(
            dataframe_with_internal_gap,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert len(gaps) == 2
        assert gaps[0] == (
            _make_datetime(2024, 1, 1, 3, 0, 0),
            _make_datetime(2024, 1, 1, 6, 0, 0),
        )
        assert gaps[1] == (_make_datetime(2024, 1, 1, 7, 0, 0), end_date)

    def test_returns_multiple_internal_gaps_when_multiple_gaps_exist(
        self, dataframe_with_multiple_gaps, start_date, end_date, timeframe
    ):
        gaps = compute_storage_gaps(
            dataframe_with_multiple_gaps,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert len(gaps) == 4
        assert gaps[0] == (
            _make_datetime(2024, 1, 1, 2, 0, 0),
            _make_datetime(2024, 1, 1, 3, 0, 0),
        )
        assert gaps[1] == (
            _make_datetime(2024, 1, 1, 4, 0, 0),
            _make_datetime(2024, 1, 1, 6, 0, 0),
        )
        assert gaps[2] == (
            _make_datetime(2024, 1, 1, 8, 0, 0),
            _make_datetime(2024, 1, 1, 10, 0, 0),
        )
        assert gaps[3] == (_make_datetime(2024, 1, 1, 10, 0, 0), end_date)


class TestComputeStorageGapsComplexScenarios:
    """Test cases for complex scenarios combining multiple gap types."""

    def test_handles_unsorted_data_correctly(
        self, dataframe_unsorted, start_date, end_date, timeframe
    ):
        gaps = compute_storage_gaps(
            dataframe_unsorted,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert len(gaps) == 3
        assert gaps[0] == (start_date, _make_datetime(2024, 1, 1, 6, 0, 0))
        assert gaps[1] == (
            _make_datetime(2024, 1, 1, 3, 0, 0),
            _make_datetime(2024, 1, 1, 8, 0, 0),
        )
        assert gaps[2] == (_make_datetime(2024, 1, 1, 4, 0, 0), end_date)

    def test_handles_data_with_duplicates(self):
        dates = [
            _make_datetime(2024, 1, 1, 2, 0, 0),
            _make_datetime(2024, 1, 1, 4, 0, 0),
            _make_datetime(2024, 1, 1, 4, 0, 0),
            _make_datetime(2024, 1, 1, 8, 0, 0),
        ]
        dataframe = pd.DataFrame(
            {"close": [2, 4, 4, 8]}, index=_make_dataframe_index(dates)
        )
        start_date = _make_datetime(2024, 1, 1, 0, 0, 0)
        end_date = _make_datetime(2024, 1, 1, 12, 0, 0)

        gaps = compute_storage_gaps(
            dataframe, start_date, end_date, "1h", market_open_mask=_every_stamp_open
        )

        assert len(gaps) == 4
        assert gaps[0] == (start_date, _make_datetime(2024, 1, 1, 2, 0, 0))
        assert gaps[1] == (
            _make_datetime(2024, 1, 1, 3, 0, 0),
            _make_datetime(2024, 1, 1, 4, 0, 0),
        )
        assert gaps[2] == (
            _make_datetime(2024, 1, 1, 5, 0, 0),
            _make_datetime(2024, 1, 1, 8, 0, 0),
        )
        assert gaps[3] == (_make_datetime(2024, 1, 1, 8, 0, 0), end_date)


class TestComputeStorageGapsEdgeCases:
    """Test cases for edge cases and boundary conditions."""

    def test_handles_single_point_at_window_start(self):
        dataframe = pd.DataFrame(
            {"close": [100.0]},
            index=_make_dataframe_index([_make_datetime(2024, 1, 1, 0, 0, 0)]),
        )
        start_date = _make_datetime(2024, 1, 1, 0, 0, 0)
        end_date = _make_datetime(2024, 1, 1, 12, 0, 0)

        gaps = compute_storage_gaps(
            dataframe, start_date, end_date, "1h", market_open_mask=_every_stamp_open
        )

        assert len(gaps) == 1
        assert gaps[0] == (start_date, end_date)

    def test_handles_single_point_at_window_end(self):
        dataframe = pd.DataFrame(
            {"close": [100.0]},
            index=_make_dataframe_index([_make_datetime(2024, 1, 1, 12, 0, 0)]),
        )
        start_date = _make_datetime(2024, 1, 1, 0, 0, 0)
        end_date = _make_datetime(2024, 1, 1, 12, 0, 0)

        gaps = compute_storage_gaps(
            dataframe, start_date, end_date, "1h", market_open_mask=_every_stamp_open
        )

        assert len(gaps) == 1
        assert gaps[0] == (start_date, end_date)

    def test_handles_window_with_single_timeframe_step(self):
        dataframe = pd.DataFrame(
            {"close": [100.0]},
            index=_make_dataframe_index([_make_datetime(2024, 1, 1, 1, 0, 0)]),
        )
        start_date = _make_datetime(2024, 1, 1, 0, 0, 0)
        end_date = _make_datetime(2024, 1, 1, 1, 0, 0)

        gaps = compute_storage_gaps(
            dataframe, start_date, end_date, "1h", market_open_mask=_every_stamp_open
        )

        assert len(gaps) == 1
        assert gaps[0] == (start_date, _make_datetime(2024, 1, 1, 1, 0, 0))

    def test_handles_exact_window_boundaries(self):
        dates = [
            _make_datetime(2024, 1, 1, 0, 0, 0),
            _make_datetime(2024, 1, 1, 12, 0, 0),
        ]
        dataframe = pd.DataFrame({"close": [1, 2]}, index=pd.DatetimeIndex(dates))
        start_date = _make_datetime(2024, 1, 1, 0, 0, 0)
        end_date = _make_datetime(2024, 1, 1, 12, 0, 0)

        gaps = compute_storage_gaps(
            dataframe, start_date, end_date, "1h", market_open_mask=_every_stamp_open
        )

        assert len(gaps) == 1
        assert gaps[0] == (
            _make_datetime(2024, 1, 1, 1, 0, 0),
            _make_datetime(2024, 1, 1, 12, 0, 0),
        )


class TestComputeStorageGapsDifferentTimeframes:
    """Test cases for different timeframe configurations."""

    @pytest.mark.parametrize(
        ("timeframe", "expected_step_ms"),
        [
            ("1m", 60_000),
            ("5m", 300_000),
            ("15m", 900_000),
            ("1h", 3_600_000),
            ("4h", 14_400_000),
            ("1d", 86_400_000),
        ],
    )
    def test_handles_different_timeframes_correctly(self, timeframe, expected_step_ms):
        start_time = _make_datetime(2024, 1, 1, 0, 0, 0)
        gap_duration_ms = expected_step_ms * 3

        second_timestamp = (start_time.timestamp() * 1000 + gap_duration_ms) / 1000
        second_time = datetime.fromtimestamp(second_timestamp, tz=pytz.UTC)

        dates = [start_time, second_time]
        dataframe = pd.DataFrame({"close": [1, 2]}, index=pd.DatetimeIndex(dates))

        end_timestamp = (start_time.timestamp() * 1000 + gap_duration_ms * 2) / 1000
        end_time = datetime.fromtimestamp(end_timestamp, tz=pytz.UTC)

        gaps = compute_storage_gaps(
            dataframe,
            start_time,
            end_time,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert len(gaps) >= 1
        assert isinstance(gaps, list)
        assert all(isinstance(gap, tuple) and len(gap) == 2 for gap in gaps)

    def test_handles_minute_timeframe_with_precise_gaps(self):
        dates = [
            _make_datetime(2024, 1, 1, 0, 0, 0),
            _make_datetime(2024, 1, 1, 0, 1, 0),
            _make_datetime(2024, 1, 1, 0, 4, 0),
            _make_datetime(2024, 1, 1, 0, 5, 0),
        ]
        dataframe = pd.DataFrame({"close": [1, 2, 3, 4]}, index=pd.DatetimeIndex(dates))
        start_date = _make_datetime(2024, 1, 1, 0, 0, 0)
        end_date = _make_datetime(2024, 1, 1, 0, 10, 0)

        gaps = compute_storage_gaps(
            dataframe, start_date, end_date, "1m", market_open_mask=_every_stamp_open
        )

        assert len(gaps) == 2
        assert gaps[0] == (
            _make_datetime(2024, 1, 1, 0, 2, 0),
            _make_datetime(2024, 1, 1, 0, 4, 0),
        )
        assert gaps[1] == (_make_datetime(2024, 1, 1, 0, 5, 0), end_date)


class TestComputeStorageGapsBoundaryConditions:
    """Test cases for boundary conditions and special scenarios."""

    def test_handles_gap_at_exact_step_boundary(self):
        dates = [
            _make_datetime(2024, 1, 1, 0, 0, 0),
            _make_datetime(2024, 1, 1, 2, 0, 0),
            _make_datetime(2024, 1, 1, 3, 0, 0),
        ]
        dataframe = pd.DataFrame({"close": [1, 2, 3]}, index=pd.DatetimeIndex(dates))
        start_date = _make_datetime(2024, 1, 1, 0, 0, 0)
        end_date = _make_datetime(2024, 1, 1, 6, 0, 0)

        gaps = compute_storage_gaps(
            dataframe, start_date, end_date, "1h", market_open_mask=_every_stamp_open
        )

        assert len(gaps) == 2
        assert gaps[0] == (
            _make_datetime(2024, 1, 1, 1, 0, 0),
            _make_datetime(2024, 1, 1, 2, 0, 0),
        )
        assert gaps[1] == (_make_datetime(2024, 1, 1, 3, 0, 0), end_date)

    def test_handles_a_window_shorter_than_one_candle(self):
        dataframe = pd.DataFrame(
            {"close": [100.0]},
            index=_make_dataframe_index([_make_datetime(2024, 1, 1, 0, 30, 0)]),
        )
        start_date = _make_datetime(2024, 1, 1, 0, 0, 0)
        end_date = _make_datetime(2024, 1, 1, 0, 45, 0)

        gaps = compute_storage_gaps(
            dataframe, start_date, end_date, "1h", market_open_mask=_every_stamp_open
        )

        assert gaps == []

    def test_returns_correct_type(
        self, empty_dataframe, start_date, end_date, timeframe
    ):
        gaps = compute_storage_gaps(
            empty_dataframe,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )

        assert isinstance(gaps, list)
        assert len(gaps) == 1
        assert isinstance(gaps[0], tuple)
        assert len(gaps[0]) == 2
        assert isinstance(gaps[0][0], datetime)
        assert isinstance(gaps[0][1], datetime)


class TestComputeStorageGapsOnMonths:
    """Months run twenty-eight to thirty-one days, so their gaps follow the calendar."""

    def test_returns_empty_when_every_month_of_the_window_is_stored(self):
        months = [_make_datetime(2024, month, 1) for month in range(1, 13)]
        dataframe = pd.DataFrame(
            {"close": range(len(months))}, index=_make_dataframe_index(months)
        )

        missing_ranges = compute_storage_gaps(
            dataframe, months[0], months[-1], "1M", market_open_mask=_every_stamp_open
        )

        assert missing_ranges == []

    def test_asks_from_the_stamp_of_a_month_missing_between_two_stored_months(self):
        months = [
            _make_datetime(2024, 1, 1),
            _make_datetime(2024, 3, 1),
            _make_datetime(2024, 4, 1),
        ]
        dataframe = pd.DataFrame(
            {"close": [1, 2, 3]}, index=_make_dataframe_index(months)
        )

        missing_ranges = compute_storage_gaps(
            dataframe, months[0], months[-1], "1M", market_open_mask=_every_stamp_open
        )

        assert missing_ranges == [
            (_make_datetime(2024, 2, 1), _make_datetime(2024, 3, 1))
        ]

    def test_asks_for_a_run_of_missing_months_as_one_range(self):
        months = [_make_datetime(2024, 1, 1), _make_datetime(2024, 6, 1)]
        dataframe = pd.DataFrame({"close": [1, 2]}, index=_make_dataframe_index(months))

        missing_ranges = compute_storage_gaps(
            dataframe, months[0], months[-1], "1M", market_open_mask=_every_stamp_open
        )

        assert missing_ranges == [
            (_make_datetime(2024, 2, 1), _make_datetime(2024, 6, 1))
        ]

    def test_returns_empty_across_a_february_of_twenty_eight_days(self):
        months = [
            _make_datetime(2025, 1, 1),
            _make_datetime(2025, 2, 1),
            _make_datetime(2025, 3, 1),
        ]
        dataframe = pd.DataFrame(
            {"close": [1, 2, 3]}, index=_make_dataframe_index(months)
        )

        missing_ranges = compute_storage_gaps(
            dataframe, months[0], months[-1], "1M", market_open_mask=_every_stamp_open
        )

        assert missing_ranges == []

    def test_returns_empty_across_a_year_end(self):
        months = [
            _make_datetime(2025, 11, 1),
            _make_datetime(2025, 12, 1),
            _make_datetime(2026, 1, 1),
        ]
        dataframe = pd.DataFrame(
            {"close": [1, 2, 3]}, index=_make_dataframe_index(months)
        )

        missing_ranges = compute_storage_gaps(
            dataframe, months[0], months[-1], "1M", market_open_mask=_every_stamp_open
        )

        assert missing_ranges == []


class TestComputeStorageGapsOnWeeks:
    def test_returns_empty_when_every_week_of_the_window_is_stored(self):
        weeks = [_make_datetime(2024, 1, day) for day in (1, 8, 15, 22, 29)]
        dataframe = pd.DataFrame(
            {"close": range(len(weeks))}, index=_make_dataframe_index(weeks)
        )

        missing_ranges = compute_storage_gaps(
            dataframe, weeks[0], weeks[-1], "1w", market_open_mask=_every_stamp_open
        )

        assert missing_ranges == []

    def test_asks_from_the_stamp_of_a_week_missing_between_two_stored_weeks(self):
        weeks = [_make_datetime(2024, 1, 1), _make_datetime(2024, 1, 15)]
        dataframe = pd.DataFrame({"close": [1, 2]}, index=_make_dataframe_index(weeks))

        missing_ranges = compute_storage_gaps(
            dataframe, weeks[0], weeks[-1], "1w", market_open_mask=_every_stamp_open
        )

        assert missing_ranges == [
            (_make_datetime(2024, 1, 8), _make_datetime(2024, 1, 15))
        ]


_SATURDAY = 5


def _weekdays_open(index: pd.DatetimeIndex) -> np.ndarray:
    return index.dayofweek < _SATURDAY


def _hourly_closes(dates: list[datetime], holes: list[datetime]) -> pd.DataFrame:
    closes = [np.nan if date in holes else 1.0 for date in dates]
    return pd.DataFrame({"close": closes}, index=_make_dataframe_index(dates))


class TestComputeStorageGapsOnStoredHoles:
    """A stored row no candle stands behind covers nothing inside market hours."""

    def test_a_hole_on_a_calendar_that_never_closes(self, start_date, end_date):
        hole = _make_datetime(2024, 1, 1, 6)
        stored = _hourly_closes(
            [_make_datetime(2024, 1, 1, hour) for hour in range(13)], [hole]
        )

        gaps = compute_storage_gaps(
            stored, start_date, end_date, "1h", market_open_mask=_every_stamp_open
        )

        assert gaps == [(hole, _make_datetime(2024, 1, 1, 7))]

    def test_a_hole_inside_market_hours(self, start_date, end_date):
        hole = _make_datetime(2024, 1, 1, 6)
        stored = _hourly_closes(
            [_make_datetime(2024, 1, 1, hour) for hour in range(13)], [hole]
        )

        gaps = compute_storage_gaps(
            stored, start_date, end_date, "1h", market_open_mask=_weekdays_open
        )

        assert gaps == [(hole, _make_datetime(2024, 1, 1, 7))]

    def test_a_hole_outside_market_hours(self):
        friday = _make_datetime(2024, 1, 5)
        monday = _make_datetime(2024, 1, 8)
        dates = [friday + timedelta(hours=hour) for hour in range(73)]
        weekend = [date for date in dates if date.weekday() >= _SATURDAY]
        stored = _hourly_closes(dates, weekend)

        gaps = compute_storage_gaps(
            stored, friday, monday, "1h", market_open_mask=_weekdays_open
        )

        assert gaps == []
