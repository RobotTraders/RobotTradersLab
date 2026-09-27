import pandas as pd
import pytest

import robottraderslab.ohlcv_provider.mock_ohlcv_provider as mod
from robottraderslab import Symbol
from robottraderslab.ohlcv_provider.mock_ohlcv_provider import MockOHLCVProvider


@pytest.fixture(scope="module")
def symbol_btc() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


@pytest.fixture(scope="module")
def timeframe_1h() -> str:
    return "1h"


def test_lookback_generates_expected_length_and_alignment(
    monkeypatch, symbol_btc, timeframe_1h
):
    fixed_now = pd.Timestamp("2024-01-01T10:37:00+00:00")

    def _fake_to_datetime(_):
        return fixed_now.to_pydatetime()

    monkeypatch.setattr(mod, "to_datetime", _fake_to_datetime, raising=True)

    provider = MockOHLCVProvider(lookback=5)
    df = provider.fetch_ohlcv(symbol_btc, timeframe_1h)

    assert len(df) == 5
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None

    assert df.index[-1] == pd.Timestamp("2024-01-01T10:00:00+00:00")
    diffs = df.index.to_series().diff().dropna()
    assert (diffs == pd.to_timedelta(1, unit="h")).all()


def test_date_window_hourly_is_inclusive_and_aligned(symbol_btc, timeframe_1h):
    provider = MockOHLCVProvider()
    provider.set_dates("2024-01-01T00:15:00+00:00", "2024-01-01T04:45:00+00:00")
    df = provider.fetch_ohlcv(symbol_btc, timeframe_1h)

    assert not df.empty
    assert df.index.min() == pd.Timestamp("2024-01-01T00:00:00+00:00")
    assert df.index.max() == pd.Timestamp("2024-01-01T04:00:00+00:00")
    assert len(df) == 5


def test_weekly_uses_7_day_intervals(symbol_btc):
    provider = MockOHLCVProvider()
    # Both dates are Wednesday
    provider.set_dates("2024-01-03T12:00:00+00:00", "2024-01-17T12:00:00+00:00")

    df = provider.fetch_ohlcv(symbol_btc, "1w")

    # Weekly uses 7-day intervals with alignment
    expected = [
        pd.Timestamp("2023-12-28T00:00:00+00:00"),
        pd.Timestamp("2024-01-04T00:00:00+00:00"),
        pd.Timestamp("2024-01-11T00:00:00+00:00"),
    ]
    assert list(df.index) == expected


def test_monthly_uses_30_day_intervals(symbol_btc):
    provider = MockOHLCVProvider()
    provider.set_dates("2024-01-15T00:00:00+00:00", "2024-03-20T00:00:00+00:00")
    df = provider.fetch_ohlcv(symbol_btc, "1M")

    # Monthly uses 30-day intervals with alignment
    expected = [
        pd.Timestamp("2023-12-19T00:00:00+00:00"),
        pd.Timestamp("2024-01-18T00:00:00+00:00"),
        pd.Timestamp("2024-02-17T00:00:00+00:00"),
        pd.Timestamp("2024-03-18T00:00:00+00:00"),
    ]
    assert list(df.index) == expected
