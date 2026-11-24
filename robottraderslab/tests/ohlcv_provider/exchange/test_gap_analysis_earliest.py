from datetime import datetime

import pandas as pd
import pytest
import pytz

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
    dt = datetime(year, month, day, hour, minute, second)
    return dt.replace(tzinfo=pytz.UTC) if is_timezone_aware() else dt


def _make_dataframe_index(dates: list[datetime]) -> pd.DatetimeIndex:
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
def timeframe() -> TimeFrame:
    return "1h"


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    return pd.DataFrame(index=pd.DatetimeIndex([]))


class TestEarliestAvailableShrinks:
    def test_earliest_available_shrinks_gap_before_first(self, timeframe: TimeFrame):
        start_date = _make_datetime(2024, 9, 1)
        end_date = _make_datetime(2024, 12, 15)
        earliest_available = _make_datetime(2024, 12, 1)
        existing_dates = [_make_datetime(2024, 12, 1, h) for h in range(24)]
        existing_data = pd.DataFrame(
            {"close": range(len(existing_dates))},
            index=_make_dataframe_index(existing_dates),
        )

        gaps = compute_storage_gaps(
            existing_data,
            start_date,
            end_date,
            timeframe,
            earliest_available=earliest_available,
            market_open_mask=_every_stamp_open,
        )

        gap_starts = [start for start, _ in gaps]
        for gap_start in gap_starts:
            assert gap_start >= earliest_available


class TestEarliestAvailableNone:
    def test_earliest_available_none_preserves_original_behavior(
        self, timeframe: TimeFrame
    ):
        start_date = _make_datetime(2024, 9, 1)
        end_date = _make_datetime(2024, 12, 15)
        existing_dates = [_make_datetime(2024, 12, 1, h) for h in range(24)]
        existing_data = pd.DataFrame(
            {"close": range(len(existing_dates))},
            index=_make_dataframe_index(existing_dates),
        )

        gaps_without = compute_storage_gaps(
            existing_data,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )
        gaps_with_none = compute_storage_gaps(
            existing_data,
            start_date,
            end_date,
            timeframe,
            earliest_available=None,
            market_open_mask=_every_stamp_open,
        )

        assert gaps_without == gaps_with_none


class TestEarliestAvailableBeforeStartDate:
    def test_earliest_available_before_start_date_is_noop(self, timeframe: TimeFrame):
        start_date = _make_datetime(2024, 6, 1)
        end_date = _make_datetime(2024, 6, 2)
        earliest_available = _make_datetime(2024, 1, 1)
        existing_dates = [_make_datetime(2024, 6, 1, h) for h in range(24)]
        existing_data = pd.DataFrame(
            {"close": range(len(existing_dates))},
            index=_make_dataframe_index(existing_dates),
        )

        gaps_without = compute_storage_gaps(
            existing_data,
            start_date,
            end_date,
            timeframe,
            market_open_mask=_every_stamp_open,
        )
        gaps_with_early = compute_storage_gaps(
            existing_data,
            start_date,
            end_date,
            timeframe,
            earliest_available=earliest_available,
            market_open_mask=_every_stamp_open,
        )

        assert gaps_without == gaps_with_early


class TestEarliestAvailableAfterEndDate:
    def test_earliest_available_after_end_date(
        self, empty_dataframe: pd.DataFrame, timeframe: TimeFrame
    ):
        start_date = _make_datetime(2024, 1, 1)
        end_date = _make_datetime(2024, 6, 1)
        earliest_available = _make_datetime(2024, 12, 1)

        gaps = compute_storage_gaps(
            empty_dataframe,
            start_date,
            end_date,
            timeframe,
            earliest_available=earliest_available,
            market_open_mask=_every_stamp_open,
        )

        assert gaps == []


class TestEarliestAvailableWithEmptyData:
    def test_earliest_available_with_empty_data(
        self, empty_dataframe: pd.DataFrame, timeframe: TimeFrame
    ):
        start_date = _make_datetime(2024, 9, 1)
        end_date = _make_datetime(2024, 12, 15)
        earliest_available = _make_datetime(2024, 11, 1)

        gaps = compute_storage_gaps(
            empty_dataframe,
            start_date,
            end_date,
            timeframe,
            earliest_available=earliest_available,
            market_open_mask=_every_stamp_open,
        )

        assert len(gaps) == 1
        gap_start, gap_end = gaps[0]
        assert gap_start == earliest_available
        assert gap_end == end_date
