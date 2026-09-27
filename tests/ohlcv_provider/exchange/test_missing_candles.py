import logging

import numpy as np
import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.ohlcv_provider.exchange.missing_candles import (
    handle_missing_candles,
)
from robottraderslab.ohlcv_provider.types import OhlcvColumn


def ohlcv_at(moments: list[str]) -> pd.DataFrame:
    stamps = [
        int(pd.Timestamp(moment, tz="UTC").timestamp() * 1000) for moment in moments
    ]
    return pd.DataFrame(
        {
            OhlcvColumn.TIMESTAMP: stamps,
            OhlcvColumn.OPEN: np.arange(100.0, 100.0 + len(stamps)),
            OhlcvColumn.HIGH: np.arange(105.0, 105.0 + len(stamps)),
            OhlcvColumn.LOW: np.arange(95.0, 95.0 + len(stamps)),
            OhlcvColumn.CLOSE: np.arange(101.0, 101.0 + len(stamps)),
            OhlcvColumn.VOLUME: np.arange(1000.0, 1000.0 + len(stamps)),
        }
    )


def moments_of(ohlcv: pd.DataFrame) -> list[str]:
    return sorted(
        pd.Timestamp(int(stamp), unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")
        for stamp in ohlcv[OhlcvColumn.TIMESTAMP]
    )


def filled_rows(ohlcv: pd.DataFrame) -> pd.DataFrame:
    return ohlcv[ohlcv[OhlcvColumn.CLOSE].isna()]


@pytest.fixture
def symbol() -> Symbol:
    return Symbol.create("BTC/USDT")


@pytest.fixture
def every_hour() -> pd.DataFrame:
    return ohlcv_at(
        [
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 02:00",
            "2024-01-01 03:00",
            "2024-01-01 04:00",
        ]
    )


@pytest.fixture
def hours_with_holes() -> pd.DataFrame:
    return ohlcv_at(
        [
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 03:00",
            "2024-01-01 05:00",
        ]
    )


class TestTimeframesOfOneLength:
    def test_an_hourly_range_without_holes(self, symbol, every_hour):
        completed = handle_missing_candles(every_hour, symbol, "1h")

        pd.testing.assert_frame_equal(completed, every_hour)

    def test_one_hour_missing_between_two_stored_hours(self, symbol):
        completed = handle_missing_candles(
            ohlcv_at(["2024-01-01 00:00", "2024-01-01 02:00"]), symbol, "1h"
        )

        assert moments_of(completed) == [
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 02:00",
        ]
        assert moments_of(filled_rows(completed)) == ["2024-01-01 01:00"]

    def test_a_filled_candle_carries_no_values(self, symbol):
        completed = handle_missing_candles(
            ohlcv_at(["2024-01-01 00:00", "2024-01-01 02:00"]), symbol, "1h"
        )

        filled = filled_rows(completed).iloc[0]
        assert pd.isna(filled[OhlcvColumn.OPEN])
        assert pd.isna(filled[OhlcvColumn.HIGH])
        assert pd.isna(filled[OhlcvColumn.LOW])
        assert pd.isna(filled[OhlcvColumn.CLOSE])
        assert pd.isna(filled[OhlcvColumn.VOLUME])

    def test_several_hours_missing(self, symbol, hours_with_holes):
        completed = handle_missing_candles(hours_with_holes, symbol, "1h")

        assert moments_of(filled_rows(completed)) == [
            "2024-01-01 02:00",
            "2024-01-01 04:00",
        ]

    @pytest.mark.parametrize(
        ("timeframe", "filled_count"), [("30m", 7), ("1h", 2), ("2h", 3)]
    )
    def test_the_same_stamps_read_on_another_timeframe(
        self, symbol, hours_with_holes, timeframe, filled_count
    ):
        completed = handle_missing_candles(hours_with_holes, symbol, timeframe)

        assert len(filled_rows(completed)) == filled_count

    def test_stored_candles_keep_their_values(self, symbol, hours_with_holes):
        completed = handle_missing_candles(hours_with_holes, symbol, "1h")

        stored = completed.dropna().set_index(OhlcvColumn.TIMESTAMP)
        expected = hours_with_holes.set_index(OhlcvColumn.TIMESTAMP)
        pd.testing.assert_frame_equal(stored, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "stamps",
        [["2024-01-01 00:00"], ["2024-01-01 00:00", "2024-01-01 01:00"]],
    )
    def test_a_range_with_nothing_missing(self, symbol, stamps):
        stored = ohlcv_at(stamps)

        completed = handle_missing_candles(stored, symbol, "1h")

        pd.testing.assert_frame_equal(completed, stored)

    def test_a_long_run_of_missing_candles(self, symbol):
        completed = handle_missing_candles(
            ohlcv_at(["2024-01-01 00:00", "2024-01-01 09:00"]), symbol, "1h"
        )

        assert len(completed) == 10
        assert len(filled_rows(completed)) == 8

    def test_filled_stamps_stay_whole_numbers(self, symbol):
        completed = handle_missing_candles(
            ohlcv_at(["2024-01-01 00:00", "2024-01-01 02:00"]), symbol, "1h"
        )

        for stamp in completed[OhlcvColumn.TIMESTAMP]:
            assert isinstance(stamp, (int, np.integer))


class TestTimeframesWhoseCandlesVary:
    def test_a_year_of_months_without_holes(self, symbol):
        every_month = ohlcv_at([f"2024-{month:02d}-01" for month in range(1, 13)])

        completed = handle_missing_candles(every_month, symbol, "1M")

        pd.testing.assert_frame_equal(completed, every_month)

    def test_a_month_missing_between_two_stored_months(self, symbol):
        completed = handle_missing_candles(
            ohlcv_at(["2024-01-01", "2024-03-01"]), symbol, "1M"
        )

        assert moments_of(completed) == [
            "2024-01-01 00:00",
            "2024-02-01 00:00",
            "2024-03-01 00:00",
        ]
        assert moments_of(filled_rows(completed)) == ["2024-02-01 00:00"]

    def test_a_run_of_months_missing(self, symbol):
        completed = handle_missing_candles(
            ohlcv_at(["2024-01-01", "2024-06-01"]), symbol, "1M"
        )

        assert moments_of(filled_rows(completed)) == [
            "2024-02-01 00:00",
            "2024-03-01 00:00",
            "2024-04-01 00:00",
            "2024-05-01 00:00",
        ]

    def test_a_february_of_twenty_nine_days(self, symbol):
        leap_february = ohlcv_at(["2024-01-01", "2024-02-01", "2024-03-01"])

        completed = handle_missing_candles(leap_february, symbol, "1M")

        pd.testing.assert_frame_equal(completed, leap_february)

    def test_a_february_of_twenty_eight_days(self, symbol):
        short_february = ohlcv_at(["2025-01-01", "2025-02-01", "2025-03-01"])

        completed = handle_missing_candles(short_february, symbol, "1M")

        pd.testing.assert_frame_equal(completed, short_february)

    def test_months_across_a_year_end(self, symbol):
        across_new_year = ohlcv_at(["2025-11-01", "2025-12-01", "2026-01-01"])

        completed = handle_missing_candles(across_new_year, symbol, "1M")

        pd.testing.assert_frame_equal(completed, across_new_year)

    def test_a_year_of_weeks_without_holes(self, symbol):
        every_week = ohlcv_at(
            pd.date_range("2024-01-01", periods=52, freq="7D")
            .strftime("%Y-%m-%d")
            .tolist()
        )

        completed = handle_missing_candles(every_week, symbol, "1w")

        pd.testing.assert_frame_equal(completed, every_week)

    def test_a_week_missing_between_two_stored_weeks(self, symbol):
        completed = handle_missing_candles(
            ohlcv_at(["2024-01-01", "2024-01-15"]), symbol, "1w"
        )

        assert moments_of(filled_rows(completed)) == ["2024-01-08 00:00"]


class TestGapWarnings:
    def test_a_range_without_holes(self, symbol, every_hour, caplog):
        with caplog.at_level(logging.WARNING):
            handle_missing_candles(every_hour, symbol, "1h")

        assert caplog.records == []

    def test_holes_inside_market_hours(self, symbol, hours_with_holes, caplog):
        with caplog.at_level(logging.WARNING):
            handle_missing_candles(hours_with_holes, symbol, "1h")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"
        assert "2 missing candles replaced with NaN" in caplog.records[0].message
        assert str(symbol) in caplog.records[0].message

    def test_holes_outside_market_hours(self, symbol, hours_with_holes, caplog):
        with caplog.at_level(logging.WARNING):
            completed = handle_missing_candles(
                hours_with_holes,
                symbol,
                "1h",
                market_open_mask=lambda index: np.zeros(len(index), dtype=bool),
            )

        assert caplog.records == []
        assert len(filled_rows(completed)) == 2

    def test_only_the_holes_inside_market_hours_are_named(
        self, symbol, hours_with_holes, caplog
    ):
        with caplog.at_level(logging.WARNING):
            handle_missing_candles(
                hours_with_holes,
                symbol,
                "1h",
                market_open_mask=lambda index: index.hour == 2,
            )

        assert "1 missing candles" in caplog.records[0].message
        assert "02:00" in caplog.records[0].message
        assert "04:00" not in caplog.records[0].message

    def test_a_mask_that_never_closes_the_market(
        self, symbol, hours_with_holes, caplog
    ):
        with caplog.at_level(logging.WARNING):
            handle_missing_candles(
                hours_with_holes, symbol, "1h", market_open_mask=lambda index: None
            )

        assert "2 missing candles" in caplog.records[0].message

    def test_consecutive_hours_are_named_as_one_range(self, symbol, caplog):
        with caplog.at_level(logging.WARNING):
            handle_missing_candles(
                ohlcv_at(["2024-01-01 00:00", "2024-01-01 04:00"]), symbol, "1h"
            )

        assert "3 missing candles" in caplog.records[0].message
        assert "2024-01-01 01:00 -> 2024-01-01 03:00" in caplog.records[0].message

    def test_consecutive_months_are_named_as_one_range(self, symbol, caplog):
        with caplog.at_level(logging.WARNING):
            handle_missing_candles(ohlcv_at(["2024-01-01", "2024-05-01"]), symbol, "1M")

        assert "3 missing candles" in caplog.records[0].message
        assert "2024-02-01 00:00 -> 2024-04-01 00:00" in caplog.records[0].message
