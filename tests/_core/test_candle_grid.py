from datetime import UTC, datetime

import pandas as pd
import pytest

from robottraderslab._core import (
    candle_stamps,
    candle_step,
    candles_back,
    last_per_month,
    to_milliseconds,
)


def stamp_ms(moment: str) -> int:
    return int(pd.Timestamp(moment, tz="UTC").timestamp() * 1000)


def as_dates(stamps) -> list[str]:
    return [
        pd.Timestamp(int(stamp), unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")
        for stamp in stamps
    ]


class TestCandleStamps:
    def test_one_stamp_per_calendar_month(self):
        monthly = candle_stamps("1M", stamp_ms("2024-01-01"), stamp_ms("2024-12-01"))

        assert len(monthly) == 12
        assert as_dates(monthly)[-1] == "2024-12-01 00:00"

    def test_months_of_a_range_crossing_a_leap_february(self):
        monthly = candle_stamps("1M", stamp_ms("2024-01-01"), stamp_ms("2024-04-01"))

        assert as_dates(monthly) == [
            "2024-01-01 00:00",
            "2024-02-01 00:00",
            "2024-03-01 00:00",
            "2024-04-01 00:00",
        ]

    def test_months_of_a_range_crossing_a_year_end(self):
        monthly = candle_stamps("1M", stamp_ms("2025-11-01"), stamp_ms("2026-02-01"))

        assert as_dates(monthly) == [
            "2025-11-01 00:00",
            "2025-12-01 00:00",
            "2026-01-01 00:00",
            "2026-02-01 00:00",
        ]

    def test_a_range_holding_a_single_month(self):
        monthly = candle_stamps("1M", stamp_ms("2024-05-01"), stamp_ms("2024-05-01"))

        assert as_dates(monthly) == ["2024-05-01 00:00"]

    def test_weeks_are_seven_days_apart(self):
        weekly = candle_stamps("1w", stamp_ms("2024-01-01"), stamp_ms("2024-01-29"))

        assert as_dates(weekly) == [
            "2024-01-01 00:00",
            "2024-01-08 00:00",
            "2024-01-15 00:00",
            "2024-01-22 00:00",
            "2024-01-29 00:00",
        ]

    def test_hours_are_evenly_spaced(self):
        hourly = candle_stamps(
            "1h", stamp_ms("2024-01-01 00:00"), stamp_ms("2024-01-01 03:00")
        )

        assert as_dates(hourly) == [
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 02:00",
            "2024-01-01 03:00",
        ]


class TestCandleStep:
    @pytest.mark.parametrize(
        ("timeframe", "span"),
        [
            ("1m", pd.Timedelta(minutes=1)),
            ("15m", pd.Timedelta(minutes=15)),
            ("4h", pd.Timedelta(hours=4)),
            ("1d", pd.Timedelta(days=1)),
            ("1w", pd.Timedelta(days=7)),
        ],
    )
    def test_a_timeframe_whose_candles_are_all_one_length(self, timeframe, span):
        assert candle_step(timeframe) == span

    @pytest.mark.parametrize(
        ("month_start", "next_month_start"),
        [
            ("2024-01-01", "2024-02-01"),
            ("2024-02-01", "2024-03-01"),
            ("2024-12-01", "2025-01-01"),
        ],
    )
    def test_a_month_step_lands_on_the_next_month(self, month_start, next_month_start):
        stepped = pd.Timestamp(month_start, tz="UTC") + candle_step("1M")

        assert stepped == pd.Timestamp(next_month_start, tz="UTC")

    @pytest.mark.parametrize(
        ("month_start", "days"),
        [
            ("2024-02-01", 29),
            ("2025-02-01", 28),
            ("2024-04-01", 30),
            ("2024-07-01", 31),
        ],
    )
    def test_a_month_step_lasts_as_long_as_its_own_month(self, month_start, days):
        opens_at = pd.Timestamp(month_start, tz="UTC")

        assert opens_at + candle_step("1M") - opens_at == pd.Timedelta(days=days)


@pytest.mark.parametrize(
    ("timeframe", "count", "moment", "reaches_back_to"),
    [
        (
            "1M",
            3,
            datetime(2026, 3, 17, 12, tzinfo=UTC),
            datetime(2025, 12, 17, 12, tzinfo=UTC),
        ),
        (
            "1M",
            1,
            datetime(2026, 1, 15, tzinfo=UTC),
            datetime(2025, 12, 15, tzinfo=UTC),
        ),
        ("1M", 2, datetime(2026, 4, 30, tzinfo=UTC), datetime(2026, 2, 28, tzinfo=UTC)),
        (
            "1h",
            3,
            datetime(2026, 3, 17, 12, tzinfo=UTC),
            datetime(2026, 3, 17, 9, tzinfo=UTC),
        ),
        ("1w", 2, datetime(2026, 3, 17, tzinfo=UTC), datetime(2026, 3, 3, tzinfo=UTC)),
    ],
)
def test_a_lookback_from_a_moment_off_the_grid(
    timeframe, count, moment, reaches_back_to
):
    assert pd.Timestamp(moment) - candles_back(timeframe, count) == reaches_back_to


@pytest.mark.parametrize(
    ("timeframe", "milliseconds"),
    [
        ("1m", 60_000),
        ("45m", 2_700_000),
        ("1d", 86_400_000),
        ("1M", 2_592_000_000),
    ],
)
def test_a_timeframe_span_in_milliseconds(timeframe, milliseconds):
    assert to_milliseconds(timeframe) == milliseconds


class TestLastPerMonth:
    def test_a_reading_on_the_first_midnight_ends_the_month_before(self):
        readings = pd.Series(
            [100.0, 110.0, 120.0],
            index=pd.to_datetime(
                ["2024-01-31 23:00", "2024-02-01 00:00", "2024-02-01 01:00"]
            ),
        )

        monthly = last_per_month(readings)

        assert monthly.to_dict() == {
            pd.Timestamp("2024-01-01"): 110.0,
            pd.Timestamp("2024-02-01"): 120.0,
        }

    def test_a_month_between_readings_holds_no_value(self):
        readings = pd.Series(
            [100.0, 110.0], index=pd.to_datetime(["2024-01-15", "2024-03-15"])
        )

        monthly = last_per_month(readings)

        assert monthly.index.tolist() == [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
            pd.Timestamp("2024-03-01"),
        ]
        assert pd.isna(monthly[pd.Timestamp("2024-02-01")])
