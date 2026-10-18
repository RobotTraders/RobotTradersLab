import numpy as np
import pandas as pd
import pytest

from robottraderslab.exchanges import OhlcvData, lacks_a_closed_candle, no_candles


def stamp(moment: str) -> int:
    return int(pd.Timestamp(moment, tz="UTC").timestamp() * 1000)


def candles(moments: list[str]) -> OhlcvData:
    return np.array(
        [[float(stamp(moment)), 1.0, 2.0, 0.5, 1.5, 10.0] for moment in moments]
    )


@pytest.fixture(autouse=True)
def _now(monkeypatch) -> None:
    monkeypatch.setattr(
        "time.time", lambda: pd.Timestamp("2024-03-10 12:30", tz="UTC").timestamp()
    )


def test_a_range_with_no_candle():
    assert lacks_a_closed_candle(no_candles(), "1h", stamp("2024-03-10 12:00"))


def test_a_range_reaching_the_last_closed_candle():
    rows = candles(["2024-03-10 10:00", "2024-03-10 11:00"])

    assert not lacks_a_closed_candle(rows, "1h", stamp("2024-03-10 12:00"))


def test_a_range_holding_the_candle_still_running():
    rows = candles(["2024-03-10 11:00", "2024-03-10 12:00"])

    assert not lacks_a_closed_candle(rows, "1h", stamp("2024-03-10 13:00"))


def test_a_range_whose_candles_stop_before_its_end():
    rows = candles(["2024-03-09 10:00", "2024-03-09 11:00"])

    assert lacks_a_closed_candle(rows, "1h", stamp("2024-03-10 12:00"))


def test_an_end_ahead_of_the_present_is_cut_at_the_present():
    rows = candles(["2024-03-10 10:00", "2024-03-10 11:00"])

    assert not lacks_a_closed_candle(rows, "1h", stamp("2024-03-12 00:00"))


def test_a_month_measured_on_the_calendar():
    rows = candles(["2024-01-01 00:00"])

    assert not lacks_a_closed_candle(rows, "1M", stamp("2024-02-29 00:00"))
    assert lacks_a_closed_candle(rows, "1M", stamp("2024-03-01 00:00"))
