from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from robottraderslab.exchanges import OhlcvData, no_candles, without_the_open_candle


def candles(moments: list[str]) -> OhlcvData:
    return np.array(
        [
            [
                pd.Timestamp(moment, tz="UTC").timestamp() * 1000,
                1.0,
                2.0,
                0.5,
                1.5,
                10.0,
            ]
            for moment in moments
        ]
    )


def moments_of(rows: OhlcvData) -> list[str]:
    return [
        pd.Timestamp(int(stamp), unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")
        for stamp in rows[:, 0]
    ]


@pytest.fixture
def clock(monkeypatch) -> Callable[[str], None]:
    def set_to(moment: str) -> None:
        monkeypatch.setattr(
            "time.time", lambda: pd.Timestamp(moment, tz="UTC").timestamp()
        )

    return set_to


class TestTimeframesOfOneLength:
    def test_the_candle_still_running(self, clock):
        clock("2024-01-01 02:30")

        kept = without_the_open_candle(
            candles(["2024-01-01 01:00", "2024-01-01 02:00"]), "1h"
        )

        assert moments_of(kept) == ["2024-01-01 01:00"]

    def test_the_candle_that_just_closed(self, clock):
        clock("2024-01-01 03:00")

        kept = without_the_open_candle(
            candles(["2024-01-01 01:00", "2024-01-01 02:00"]), "1h"
        )

        assert moments_of(kept) == ["2024-01-01 01:00", "2024-01-01 02:00"]

    def test_no_candles_at_all(self, clock):
        clock("2024-01-01 03:00")

        kept = without_the_open_candle(no_candles(), "1h")

        assert kept.size == 0


class TestTimeframesWhoseCandlesVary:
    def test_the_month_still_running(self, clock):
        clock("2024-02-15 00:00")

        kept = without_the_open_candle(candles(["2024-01-01", "2024-02-01"]), "1M")

        assert moments_of(kept) == ["2024-01-01 00:00"]

    @pytest.mark.parametrize("year", [2024, 2025])
    def test_a_february_the_day_it_closed(self, clock, year):
        clock(f"{year}-03-01 00:00")
        stored = [f"{year}-01-01", f"{year}-02-01"]

        kept = without_the_open_candle(candles(stored), "1M")

        assert moments_of(kept) == [f"{year}-01-01 00:00", f"{year}-02-01 00:00"]

    def test_a_month_of_thirty_one_days_a_day_before_it_closes(self, clock):
        clock("2024-01-31 00:00")

        kept = without_the_open_candle(candles(["2023-12-01", "2024-01-01"]), "1M")

        assert moments_of(kept) == ["2023-12-01 00:00"]


class TestWeeks:
    def test_the_week_that_just_closed(self, clock):
        clock("2024-01-08 00:00")

        kept = without_the_open_candle(candles(["2023-12-25", "2024-01-01"]), "1w")

        assert moments_of(kept) == ["2023-12-25 00:00", "2024-01-01 00:00"]

    def test_the_week_still_running(self, clock):
        clock("2024-01-05 00:00")

        kept = without_the_open_candle(candles(["2023-12-25", "2024-01-01"]), "1w")

        assert moments_of(kept) == ["2023-12-25 00:00"]
