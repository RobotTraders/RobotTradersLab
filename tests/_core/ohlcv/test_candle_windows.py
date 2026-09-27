from datetime import UTC, datetime

import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab._core import candle_windows
from robottraderslab._core.ohlcv.candle_windows import CandleWindow
from robottraderslab.strategies import OHLCVs, StrategyRequirements

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")


def _candles(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0] * len(timestamps),
            "high": [110.0] * len(timestamps),
            "low": [90.0] * len(timestamps),
            "close": [105.0] * len(timestamps),
            "volume": [1000.0] * len(timestamps),
        },
        index=pd.to_datetime(timestamps, utc=True),
    )


@pytest.fixture
def requirements() -> StrategyRequirements:
    return StrategyRequirements()


def test_window_spans_the_closed_candle(requirements):
    requirements.ohlcv.add(BTC, "15m")
    ohlcvs = OHLCVs({"15m": {BTC: _candles(["2026-01-01 09:15", "2026-01-01 09:30"])}})

    windows = candle_windows(requirements.ohlcv, ohlcvs, {"15m"})

    assert windows[BTC] == CandleWindow(
        start=datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
        end=datetime(2026, 1, 1, 9, 45, tzinfo=UTC),
    )


def test_naive_candle_stamps_are_read_as_utc(requirements):
    requirements.ohlcv.add(BTC, "1d")
    frame = _candles(["2026-01-01"])
    frame.index = frame.index.tz_localize(None)
    ohlcvs = OHLCVs({"1d": {BTC: frame}})

    windows = candle_windows(requirements.ohlcv, ohlcvs, {"1d"})

    assert windows[BTC].start == datetime(2026, 1, 1, tzinfo=UTC)


def test_each_symbol_takes_the_window_of_its_own_timeframe(requirements):
    requirements.ohlcv.add(BTC, "15m")
    requirements.ohlcv.add(ETH, "1h")
    ohlcvs = OHLCVs(
        {
            "15m": {BTC: _candles(["2026-01-01 09:30"])},
            "1h": {ETH: _candles(["2026-01-01 09:00"])},
        }
    )

    windows = candle_windows(requirements.ohlcv, ohlcvs, {"15m", "1h"})

    assert windows[BTC].end == datetime(2026, 1, 1, 9, 45, tzinfo=UTC)
    assert windows[ETH].end == datetime(2026, 1, 1, 10, 0, tzinfo=UTC)


def test_a_symbol_on_two_timeframes_takes_the_shortest(requirements):
    requirements.ohlcv.add(BTC, "15m")
    requirements.ohlcv.add(BTC, "1h")
    ohlcvs = OHLCVs(
        {
            "15m": {BTC: _candles(["2026-01-01 09:30"])},
            "1h": {BTC: _candles(["2026-01-01 09:00"])},
        }
    )

    windows = candle_windows(requirements.ohlcv, ohlcvs, {"15m", "1h"})

    assert windows[BTC] == CandleWindow(
        start=datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
        end=datetime(2026, 1, 1, 9, 45, tzinfo=UTC),
    )


def test_symbols_of_other_timeframes_are_left_out(requirements):
    requirements.ohlcv.add(BTC, "15m")
    requirements.ohlcv.add(ETH, "1h")
    ohlcvs = OHLCVs(
        {
            "15m": {BTC: _candles(["2026-01-01 09:30"])},
            "1h": {ETH: _candles(["2026-01-01 09:00"])},
        }
    )

    windows = candle_windows(requirements.ohlcv, ohlcvs, {"15m"})

    assert list(windows) == [BTC]


def test_a_candle_contains_only_its_own_span():
    window = CandleWindow(
        start=datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
        end=datetime(2026, 1, 1, 9, 45, tzinfo=UTC),
    )

    assert window.contains(datetime(2026, 1, 1, 9, 30, tzinfo=UTC))
    assert window.contains(datetime(2026, 1, 1, 9, 44, 59, tzinfo=UTC))
    assert not window.contains(datetime(2026, 1, 1, 9, 45, tzinfo=UTC))
    assert not window.contains(datetime(2026, 1, 1, 9, 29, 59, tzinfo=UTC))


@pytest.mark.parametrize(
    ("month_start", "next_month_start"),
    [
        ("2026-01-01", datetime(2026, 2, 1, tzinfo=UTC)),
        ("2026-02-01", datetime(2026, 3, 1, tzinfo=UTC)),
        ("2026-04-01", datetime(2026, 5, 1, tzinfo=UTC)),
        ("2026-12-01", datetime(2027, 1, 1, tzinfo=UTC)),
    ],
)
def test_a_monthly_window_ends_where_the_next_month_opens(
    requirements, month_start, next_month_start
):
    requirements.ohlcv.add(BTC, "1M")
    ohlcvs = OHLCVs({"1M": {BTC: _candles([month_start])}})

    windows = candle_windows(requirements.ohlcv, ohlcvs, {"1M"})

    assert windows[BTC].end == next_month_start


def test_a_weekly_window_spans_seven_days(requirements):
    requirements.ohlcv.add(BTC, "1w")
    ohlcvs = OHLCVs({"1w": {BTC: _candles(["2026-01-05"])}})

    windows = candle_windows(requirements.ohlcv, ohlcvs, {"1w"})

    assert windows[BTC].end == datetime(2026, 1, 12, tzinfo=UTC)
