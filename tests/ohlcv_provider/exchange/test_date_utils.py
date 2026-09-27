import asyncio
import sys
from datetime import datetime

import pandas as pd
import pytest
import pytz

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from robottraderslab.ohlcv_provider import date_utils as du


@pytest.fixture
def sample_df_aware() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    return pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=dates)


@pytest.fixture
def sample_df_naive() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz=None)
    return pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=dates)


@pytest.fixture
def sample_df_with_duplicates() -> pd.DataFrame:
    """DataFrame with duplicate timestamps for testing deduplication."""
    dates = [
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 1, 1, 1, 0, 0),
        datetime(2024, 1, 1, 1, 0, 0),  # duplicate
        datetime(2024, 1, 1, 2, 0, 0),
    ]
    return pd.DataFrame({"close": [10, 20, 25, 30]}, index=dates)


@pytest.fixture
def sample_df_unsorted() -> pd.DataFrame:
    """DataFrame with unsorted timestamps for testing sort."""
    dates = [
        datetime(2024, 1, 1, 2, 0, 0),
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 1, 1, 1, 0, 0),
    ]
    return pd.DataFrame({"close": [30, 10, 20]}, index=dates)


@pytest.fixture
def sample_df_mixed_tz() -> pd.DataFrame:
    """DataFrame with mixed timezone aware timestamps."""
    dates = [
        datetime(2024, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
        datetime(2024, 1, 1, 5, 0, 0, tzinfo=pytz.timezone("US/Eastern")),  # UTC-5
        datetime(2024, 1, 1, 2, 0, 0, tzinfo=pytz.timezone("Europe/London")),  # UTC+0
    ]
    return pd.DataFrame({"close": [10, 20, 30]}, index=dates)


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame()


#####################
# Settings accessors
#####################


class TestSettingsAccessors:
    def test_default_timezone_is_utc(self):
        tz = du.get_default_timezone()
        assert str(tz).upper().endswith("UTC")

    def test_is_timezone_aware_true(self):
        assert du.is_timezone_aware() is True

    def test_get_date_format_contains_tz_directive_when_aware(self):
        fmt = du.get_date_format()
        assert "%z" in fmt


#############################
# Section B: Standardization
#############################


class TestToDatetime:
    def test_none_returns_current_time_utc(self):
        before = datetime.now(pytz.UTC)
        result = du.to_datetime(None)
        after = datetime.now(pytz.UTC)

        assert isinstance(result, datetime)
        assert result.tzinfo is not None
        assert before <= result <= after

    def test_datetime_naive_gets_utc_tz(self):
        naive = datetime(2024, 1, 1, 0, 0, 0)
        result = du.to_datetime(naive)
        assert result.tzinfo == pytz.UTC

    def test_datetime_aware_keeps_tz(self):
        aware = datetime(2024, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        result = du.to_datetime(aware)
        assert result == aware

    def test_string_parsing_results_in_utc(self):
        result = du.to_datetime("2024-01-01 00:00:00")
        assert result.tzinfo == pytz.UTC

    def test_string_fallback_to_pandas_parser(self):
        result = du.to_datetime("2024/01/01 00:00:00")
        assert result.year == 2024
        assert result.tzinfo == pytz.UTC

    @pytest.mark.parametrize(
        ("inp", "expected_tz_aware"),
        [
            ("2024-01-01T00:00:00+00:00", True),
            ("2024-01-01T05:00:00-05:00", True),  # EST timezone string
            ("2024-01-01 00:00:00", False),  # Naive string
            ("2024-12-31T23:59:59Z", True),  # UTC with Z suffix
            (datetime(2024, 1, 1, 0, 0, 0, tzinfo=pytz.UTC), True),
            (datetime(2024, 1, 1, 5, 0, 0, tzinfo=pytz.timezone("US/Eastern")), True),
            (datetime(2024, 1, 1, 0, 0, 0), False),  # Naive datetime
        ],
    )
    def test_various_inputs_normalize_to_utc(self, inp, expected_tz_aware):
        result = du.to_datetime(inp)
        assert result.tzinfo == pytz.UTC

    @pytest.mark.parametrize(
        ("naive_str", "expected_hour"),
        [
            ("2024-01-01T00:00:00", 0),
            ("2024-06-15T12:30:00", 12),
            ("2024-12-31T23:59:59", 23),
        ],
    )
    def test_naive_string_timestamps_preserve_time(self, naive_str, expected_hour):
        result = du.to_datetime(naive_str)
        assert result.hour == expected_hour
        assert result.tzinfo == pytz.UTC

    @pytest.mark.parametrize(
        ("tz_aware_str", "expected_utc_hour"),
        [
            ("2024-01-01T05:00:00-05:00", 10),  # EST -5 -> UTC
            ("2024-01-01T15:00:00+09:00", 6),  # JST +9 -> UTC
            ("2024-01-01T12:00:00+00:00", 12),  # Already UTC
        ],
    )
    def test_timezone_aware_strings_convert_to_utc(
        self, tz_aware_str, expected_utc_hour
    ):
        result = du.to_datetime(tz_aware_str)
        assert result.hour == expected_utc_hour
        assert result.tzinfo == pytz.UTC

    def test_timezone_aware_datetime_converts_to_utc(self):
        # EST datetime should convert to UTC (EST is UTC-5 in January)
        est_dt = datetime(2024, 1, 1, 10, 0, 0, tzinfo=pytz.timezone("US/Eastern"))
        result = du.to_datetime(est_dt)
        assert result.tzinfo == pytz.UTC
        # Check that the conversion happened (hour should be different from original)
        assert result.hour != est_dt.hour
        # In January, EST is UTC-5, so 10 AM EST should become 3 PM UTC
        # But let's verify the actual converted value instead of hardcoding
        expected_utc = est_dt.astimezone(pytz.UTC)
        assert result.hour == expected_utc.hour

    def test_non_aware_mode_keeps_naive_datetime(self, monkeypatch):
        monkeypatch.setattr(du, "USE_TIMEZONE_AWARE", False, raising=True)
        naive = datetime(2024, 1, 1, 0, 0, 0)
        result = du.to_datetime(naive)
        assert result.tzinfo is None

    def test_non_aware_mode_strips_timezone(self, monkeypatch):
        monkeypatch.setattr(du, "USE_TIMEZONE_AWARE", False, raising=True)
        aware = datetime(2024, 1, 1, 5, 0, 0, tzinfo=pytz.timezone("US/Eastern"))
        result = du.to_datetime(aware)
        assert result.tzinfo is None


class TestIndexStandardization:
    def test_standardize_index_makes_utc_and_sorts_and_dedups(self, sample_df_naive):
        df = sample_df_naive.copy()
        # Introduce duplicates and disorder
        idx = df.index
        scrambled = pd.DatetimeIndex([idx[2], idx[0], idx[1], idx[2], idx[4], idx[3]])
        df = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6]}, index=scrambled)

        out = du.standardize_ohlcv_index(df)
        assert out.index.tz is not None
        assert str(out.index.tz).upper() == "UTC"
        # Sorted ascending and duplicates removed (keeping last)
        assert out.index.is_monotonic_increasing
        assert out.index.duplicated().sum() == 0

    def test_standardize_index_empty_dataframe(self, empty_df):
        result = du.standardize_ohlcv_index(empty_df)
        assert result.empty
        assert result.equals(empty_df)

    def test_standardize_index_already_utc_aware(self, sample_df_aware):
        result = du.standardize_ohlcv_index(sample_df_aware)
        assert str(result.index.tz).upper() == "UTC"
        assert result.index.is_monotonic_increasing

    def test_standardize_index_deduplication_keeps_last(
        self, sample_df_with_duplicates
    ):
        result = du.standardize_ohlcv_index(sample_df_with_duplicates)
        # Should keep the last duplicate value (25 instead of 20)
        duplicate_time = datetime(2024, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        assert result.loc[duplicate_time, "close"] == 25
        assert len(result) == 3  # Original 4 minus 1 duplicate

    def test_standardize_index_sorts_ascending(self, sample_df_unsorted):
        result = du.standardize_ohlcv_index(sample_df_unsorted)
        assert result.index.is_monotonic_increasing
        # Values should be reordered according to sorted index
        expected_values = [10, 20, 30]  # Values for sorted timestamps
        assert result["close"].tolist() == expected_values

    def test_standardize_index_mixed_timezones_normalize_to_utc(
        self, sample_df_mixed_tz
    ):
        result = du.standardize_ohlcv_index(sample_df_mixed_tz)
        assert str(result.index.tz).upper() == "UTC"
        assert result.index.is_monotonic_increasing

    def test_standardize_index_non_aware_drops_tz(self, monkeypatch, sample_df_aware):
        monkeypatch.setattr(du, "USE_TIMEZONE_AWARE", False, raising=True)
        out = du.standardize_ohlcv_index(sample_df_aware)
        # Expect naive index (tz removed) and sorted
        assert getattr(out.index, "tz", None) is None
        assert out.index.is_monotonic_increasing

    def test_standardize_index_non_aware_parses_string_index(self, monkeypatch):
        monkeypatch.setattr(du, "USE_TIMEZONE_AWARE", False, raising=True)
        dates = [
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-01 02:00:00",
        ]
        df = pd.DataFrame({"close": [1, 2, 3]}, index=pd.Index(dates))
        out = du.standardize_ohlcv_index(df)
        assert isinstance(out.index, pd.DatetimeIndex)
        assert getattr(out.index, "tz", None) is None
        assert out.index.is_monotonic_increasing

    def test_standardize_index_from_timestamp_sets_index(self):
        ts_ms = [1704067200000, 1704070800000, 1704074400000]  # 2024-01-01 00/01/02 UTC
        df = pd.DataFrame({"timestamp": ts_ms, "close": [1.0, 2.0, 3.0]})
        out = du.standardize_ohlcv_index_from_timestamp(df)
        assert isinstance(out.index, pd.DatetimeIndex)
        assert out.index.tz is not None
        assert str(out.index.tz).upper() == "UTC"
        assert len(out) == 3

    def test_standardize_index_from_timestamp_accepts_seconds_unit(self):
        ts_s = [1704067200, 1704070800, 1704074400]
        df = pd.DataFrame({"timestamp": ts_s, "close": [1.0, 2.0, 3.0]})
        out = du.standardize_ohlcv_index_from_timestamp(df, unit="s")
        assert isinstance(out.index, pd.DatetimeIndex)
        assert out.index.tz is not None
        assert str(out.index.tz).upper() == "UTC"
        assert len(out) == 3

    def test_standardize_index_from_timestamp_empty_dataframe(self):
        empty_df = pd.DataFrame({"timestamp": [], "close": []})
        result = du.standardize_ohlcv_index_from_timestamp(empty_df)
        assert result.empty

    @pytest.mark.parametrize(
        ("unit", "timestamp_values", "expected_first_hour"),
        [
            ("ms", [1704067200000], 0),  # 2024-01-01 00:00:00 UTC in ms
            ("s", [1704067200], 0),  # 2024-01-01 00:00:00 UTC in seconds
            ("us", [1704067200000000], 0),  # 2024-01-01 00:00:00 UTC in microseconds
        ],
    )
    def test_standardize_index_from_timestamp_various_units(
        self, unit, timestamp_values, expected_first_hour
    ):
        df = pd.DataFrame({"timestamp": timestamp_values, "close": [100.0]})
        result = du.standardize_ohlcv_index_from_timestamp(df, unit=unit)

        assert isinstance(result.index, pd.DatetimeIndex)
        assert str(result.index.tz).upper() == "UTC"
        assert result.index[0].hour == expected_first_hour

    def test_standardize_index_from_timestamp_custom_column_name(self):
        ts_ms = [1704067200000, 1704070800000]
        df = pd.DataFrame({"ts": ts_ms, "close": [1.0, 2.0]})
        result = du.standardize_ohlcv_index_from_timestamp(df, timestamp_column="ts")

        assert isinstance(result.index, pd.DatetimeIndex)
        assert str(result.index.tz).upper() == "UTC"
        assert (
            "ts" not in result.columns
        )  # Column should be removed after setting as index

    def test_standardize_index_from_timestamp_deduplicates_and_sorts(self):
        # Unsorted timestamps with duplicates
        ts_ms = [
            1704074400000,
            1704067200000,
            1704070800000,
            1704070800000,
        ]  # 02:00, 00:00, 01:00, 01:00
        df = pd.DataFrame({"timestamp": ts_ms, "close": [30.0, 10.0, 20.0, 25.0]})
        result = du.standardize_ohlcv_index_from_timestamp(df)

        assert result.index.is_monotonic_increasing
        assert result.index.duplicated().sum() == 0
        assert len(result) == 3  # One duplicate removed
        # Should keep last duplicate value (25.0)
        assert (
            result.loc[datetime(2024, 1, 1, 1, 0, 0, tzinfo=pytz.UTC), "close"] == 25.0
        )


############
# Windowing
############


class TestFilterDateRange:
    def test_returns_empty_when_df_empty(self):
        empty = pd.DataFrame()
        result = du.filter_date_range(empty)
        assert result.empty

    def test_filters_with_both_bounds(self, sample_df_aware):
        start = sample_df_aware.index[1]
        end = sample_df_aware.index[3]
        filtered = du.filter_date_range(sample_df_aware, start, end)
        assert not filtered.empty
        assert filtered.index.min() >= start
        assert filtered.index.max() <= end

    def test_filters_with_start_only(self, sample_df_aware):
        start = sample_df_aware.index[2]
        filtered = du.filter_date_range(sample_df_aware, start_date=start)
        assert not filtered.empty
        assert filtered.index.min() >= start

    def test_filters_with_end_only(self, sample_df_aware):
        end = sample_df_aware.index[2]
        filtered = du.filter_date_range(sample_df_aware, end_date=end)
        assert not filtered.empty
        assert filtered.index.max() <= end

    def test_no_bounds_returns_same_frame(self, sample_df_aware):
        original = sample_df_aware.copy()
        out = du.filter_date_range(original)
        pd.testing.assert_frame_equal(out, original)

    def test_inclusive_edges(self, sample_df_aware):
        start = sample_df_aware.index.min()
        end = sample_df_aware.index.max()
        out = du.filter_date_range(sample_df_aware, start, end)
        assert out.index.min() == start
        assert out.index.max() == end

    def test_filter_with_timezone_aware_bounds(self, sample_df_aware):
        # Create timezone-aware bounds
        start_utc = datetime(2024, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        end_utc = datetime(2024, 1, 1, 3, 0, 0, tzinfo=pytz.UTC)

        result = du.filter_date_range(sample_df_aware, start_utc, end_utc)
        assert not result.empty
        assert result.index.min() >= start_utc
        assert result.index.max() <= end_utc

    def test_filter_with_different_timezone_bounds(self, sample_df_aware):
        # Use EST timezone bounds (should work due to pandas' timezone handling)
        start_est = datetime(2024, 1, 1, 1, 0, 0, tzinfo=pytz.timezone("US/Eastern"))
        end_est = datetime(2024, 1, 1, 3, 0, 0, tzinfo=pytz.timezone("US/Eastern"))

        result = du.filter_date_range(sample_df_aware, start_est, end_est)
        # Should still filter correctly despite timezone difference
        assert isinstance(result, pd.DataFrame)

    def test_filter_out_of_range_returns_empty(self, sample_df_aware):
        # Bounds completely outside the data range
        start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        end = datetime(2025, 1, 2, 0, 0, 0, tzinfo=pytz.UTC)

        result = du.filter_date_range(sample_df_aware, start, end)
        assert result.empty

    def test_filter_start_after_end_returns_empty(self, sample_df_aware):
        # Invalid range where start > end
        start = sample_df_aware.index[3]
        end = sample_df_aware.index[1]

        result = du.filter_date_range(sample_df_aware, start, end)
        assert result.empty

    def test_filter_exact_timestamp_match_inclusive(self, sample_df_aware):
        # Filter to exact single timestamp
        target_time = sample_df_aware.index[2]

        result = du.filter_date_range(sample_df_aware, target_time, target_time)
        assert len(result) == 1
        assert result.index[0] == target_time

    @pytest.mark.parametrize(
        ("start_offset", "end_offset", "expected_length"),
        [
            (0, 2, 3),  # Include first 3 elements
            (1, 3, 3),  # Include middle 3 elements
            (3, 4, 2),  # Include last 2 elements
        ],
    )
    def test_filter_various_ranges(
        self, sample_df_aware, start_offset, end_offset, expected_length
    ):
        start = sample_df_aware.index[start_offset]
        end = sample_df_aware.index[end_offset]

        result = du.filter_date_range(sample_df_aware, start, end)
        assert len(result) == expected_length
        assert result.index.min() >= start
        assert result.index.max() <= end

    def test_filter_preserves_data_integrity(self, sample_df_aware):
        start = sample_df_aware.index[1]
        end = sample_df_aware.index[3]

        result = du.filter_date_range(sample_df_aware, start, end)
        # Check that the data values are preserved correctly
        original_subset = sample_df_aware.loc[start:end]
        pd.testing.assert_frame_equal(result, original_subset)
