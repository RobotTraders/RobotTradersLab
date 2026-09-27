import io

import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.ohlcv_provider.csv_ohlcv_provider import CSVOHLCVProvider


@pytest.fixture(scope="module")
def symbol_btc() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


@pytest.fixture(scope="module")
def timeframe_1h() -> str:
    return "1h"


@pytest.fixture
def dirty_csv() -> io.StringIO:
    """Create a CSV with unsorted rows and duplicate timestamps (keep last)."""
    # Intentionally unsorted with duplicate 01:00 appearing twice; last occurrence should win
    return io.StringIO("""\
date,open,high,low,close,volume
2024-01-01T01:00:00+00:00, 101, 101, 100, 101, 10
2024-01-01T00:00:00+00:00, 100, 100, 99, 100, 10
2024-01-01T01:00:00+00:00, 111, 111, 110, 111, 10
2024-01-01T02:00:00+00:00, 102, 102, 101, 102, 10
""")


def test_load_normalizes_and_de_dupes_end_to_end(dirty_csv, symbol_btc, timeframe_1h):
    provider = CSVOHLCVProvider(
        file=dirty_csv,
        symbol=str(symbol_btc),
        timeframe=timeframe_1h,
    )
    df = provider.fetch_ohlcv(symbol_btc, timeframe_1h)

    # Sorted ascending, unique timestamps (01:00 duplicate removed keeping last)
    assert not df.empty
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    assert len(df) == 3

    # Verify last duplicate kept (close at 01:00 equals 111)
    ts_0100 = pd.Timestamp("2024-01-01T01:00:00+00:00")
    # index should be tz-aware UTC per provider settings
    ts_0100 = ts_0100.tz_localize("UTC") if ts_0100.tz is None else ts_0100
    assert float(df.loc[ts_0100, "close"]) == 111.0


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [
        (
            "2024-01-01 00:00:00",
            "2024-01-01T02:00:00+00:00",
        ),  # naive start, tz-aware end
        (
            "2024-01-01T00:00:00+00:00",
            "2024-01-01 02:00:00",
        ),  # tz-aware start, naive end
    ],
)
def test_inclusive_filtering_mixed_tz_end_to_end(
    dirty_csv, symbol_btc, timeframe_1h, start_date, end_date
):
    provider = CSVOHLCVProvider(
        file=dirty_csv,
        symbol=str(symbol_btc),
        timeframe=timeframe_1h,
    )
    provider.set_dates(start_date, end_date)
    df = provider.fetch_ohlcv(symbol_btc, timeframe_1h)

    # Full window [00:00, 02:00] inclusive should return all 3 normalized rows
    assert not df.empty
    assert df.index.min() >= pd.Timestamp("2024-01-01T00:00:00+00:00", tz="UTC")
    assert df.index.max() <= pd.Timestamp("2024-01-01T02:00:00+00:00", tz="UTC")
    assert len(df) == 3
