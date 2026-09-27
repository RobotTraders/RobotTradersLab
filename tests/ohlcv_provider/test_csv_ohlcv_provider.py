import io
from collections.abc import Callable

import pandas as pd
import pytest

import robottraderslab.ohlcv_provider.csv_ohlcv_provider as provider_mod
from robottraderslab import Symbol
from robottraderslab._core import DataError, OhlcvValidationError
from robottraderslab.ohlcv_provider.csv_ohlcv_provider import CSVOHLCVProvider


@pytest.fixture
def symbol_str() -> str:
    return "BTC/USDT:USDT"


@pytest.fixture
def symbol_obj() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def timeframe_str() -> str:
    return "1h"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01T00:00:00+00:00", periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": range(10),
            "high": range(10),
            "low": range(10),
            "close": range(10),
            "volume": range(10),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def write_csv(tmp_path) -> Callable[[pd.DataFrame, str], str]:
    def _writer(df: pd.DataFrame, filename: str = "ohlcv.csv") -> str:
        csv_file = tmp_path / filename
        df.to_csv(csv_file)
        return str(csv_file)

    return _writer


def test_raises_when_csv_missing(symbol_str, timeframe_str):
    missing_path = "non_existent_file_path.csv"
    provider = CSVOHLCVProvider(
        file=missing_path,
        symbol=symbol_str,
        timeframe=timeframe_str,
    )

    with pytest.raises(
        DataError, match="No such file or directory: 'non_existent_file_path.csv'"
    ):
        provider.fetch_ohlcv(Symbol.create(symbol_str), timeframe_str)


def test_raises_when_csv_empty(tmp_path, symbol_str, timeframe_str):
    csv_file = tmp_path / "empty.csv"
    empty_df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    empty_df.to_csv(csv_file, index=False)

    with pytest.raises(DataError, match="No data available in CSV file"):
        CSVOHLCVProvider(
            file=csv_file,
            symbol=symbol_str,
            timeframe=timeframe_str,
        ).fetch_ohlcv(Symbol.create(symbol_str), timeframe_str)


def test_filters_by_date_range_returns_subset(
    write_csv, sample_df, symbol_str, symbol_obj, timeframe_str
):
    csv_file = write_csv(sample_df)

    # Choose an interior inclusive slice [idx2, idx7]
    start_dt = sample_df.index[2]
    end_dt = sample_df.index[7]
    provider = CSVOHLCVProvider(
        file=csv_file,
        symbol=symbol_str,
        timeframe=timeframe_str,
    )
    provider.set_dates(start_dt.isoformat(), end_dt.isoformat())

    df = provider.fetch_ohlcv(symbol_obj, timeframe_str)

    assert not df.empty
    assert df.index.min() >= start_dt
    assert df.index.max() <= end_dt
    # Inclusive bounds: ensure edge timestamps are present
    assert start_dt in df.index
    assert end_dt in df.index


def test_raises_when_no_data_in_date_window(
    write_csv, sample_df, symbol_str, timeframe_str
):
    csv_file = write_csv(sample_df)

    # Window strictly before available data
    before_start = (sample_df.index[0] - pd.Timedelta(hours=5)).isoformat()
    before_end = (sample_df.index[0] - pd.Timedelta(hours=1)).isoformat()

    provider = CSVOHLCVProvider(
        file=csv_file,
        symbol=symbol_str,
        timeframe=timeframe_str,
    )
    provider.set_dates(before_start, before_end)
    with pytest.raises(DataError, match="No data found between"):
        provider.fetch_ohlcv(Symbol.create(symbol_str), timeframe_str)


@pytest.mark.parametrize(
    ("fetch_symbol", "fetch_timeframe"),
    [
        (Symbol.create("ETH/USDT:USDT"), "1h"),  # mismatched symbol
        (Symbol.create("BTC/USDT:USDT"), "1d"),  # mismatched timeframe
    ],
)
def test_rejects_mismatched_symbol_or_timeframe(
    write_csv, sample_df, symbol_str, timeframe_str, fetch_symbol, fetch_timeframe
):
    csv_file = write_csv(sample_df)
    provider = CSVOHLCVProvider(
        file=csv_file,
        symbol=symbol_str,
        timeframe=timeframe_str,
    )

    with pytest.raises(OhlcvValidationError, match="single symbol/timeframe"):
        provider.fetch_ohlcv(fetch_symbol, fetch_timeframe)


def test_caches_loaded_dataframe(
    write_csv, sample_df, symbol_str, symbol_obj, timeframe_str
):
    csv_file = write_csv(sample_df)
    provider = CSVOHLCVProvider(
        file=csv_file,
        symbol=symbol_str,
        timeframe=timeframe_str,
    )

    df1 = provider.fetch_ohlcv(symbol_obj, timeframe_str)
    df2 = provider.fetch_ohlcv(symbol_obj, timeframe_str)
    assert df1 is df2


def test_cache_prevents_re_read_after_file_change(
    monkeypatch, write_csv, sample_df, symbol_str, symbol_obj, timeframe_str
):
    csv_file = write_csv(sample_df)

    call_count = {"n": 0}
    _orig_read_csv = provider_mod.pd.read_csv

    def _wrapped_read_csv(*args, **kwargs):
        call_count["n"] += 1
        return _orig_read_csv(*args, **kwargs)

    monkeypatch.setattr(provider_mod.pd, "read_csv", _wrapped_read_csv, raising=True)

    provider = CSVOHLCVProvider(
        file=csv_file,
        symbol=symbol_str,
        timeframe=timeframe_str,
    )

    df1 = provider.fetch_ohlcv(symbol_obj, timeframe_str)

    mutated = sample_df.copy()
    mutated.iloc[0, mutated.columns.get_loc("close")] = 9999  # change value
    mutated.index.name = "date"
    import pathlib

    pathlib.Path(csv_file).unlink(missing_ok=True)
    mutated.to_csv(csv_file)

    df2 = provider.fetch_ohlcv(symbol_obj, timeframe_str)

    assert df1 is df2
    assert call_count["n"] == 1


def test_accepts_a_csv_buffer(symbol_str, symbol_obj):
    csv_buffer = io.StringIO("""\
date,open,high,low,close,volume
2024-01-01T00:00:00+00:00,100,100,99,100,10
2024-01-02T00:00:00+00:00,100,100,99,100,10
""")
    provider = CSVOHLCVProvider(
        file=csv_buffer,
        symbol=symbol_str,
        timeframe="1d",
    )

    df = provider.fetch_ohlcv(symbol_obj, "1d")

    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("2024-01-01 00:00:00+0000", tz="UTC")
    assert df.index[1] == pd.Timestamp("2024-01-02 00:00:00+0000", tz="UTC")


def test_get_all_cached_ohlcv_with_mismatched_symbol():
    provider = CSVOHLCVProvider(
        file="dummy.csv",
        symbol="BTC/USDT:USDT",
        timeframe="1h",
    )
    wrong_symbol = Symbol.create("ETH/USDT:USDT")

    with pytest.raises(OhlcvValidationError, match="CSV provider configured for"):
        provider.get_all_cached_ohlcv_for_symbol(wrong_symbol)


def test_get_all_cached_ohlcv_with_matching_symbol(
    write_csv, sample_df, symbol_str, symbol_obj, timeframe_str
):
    csv_file = write_csv(sample_df)
    provider = CSVOHLCVProvider(
        file=csv_file,
        symbol=symbol_str,
        timeframe=timeframe_str,
    )

    cached_ohlcv = provider.get_all_cached_ohlcv_for_symbol(symbol_obj)

    assert not cached_ohlcv.empty
    assert len(cached_ohlcv) == len(sample_df)


def test_lookback_counts_rows_the_file_skips_a_weekend_for(
    write_csv, symbol_str, symbol_obj, timeframe_str
):
    weekday_stamps = pd.date_range(
        "2024-01-01T00:00:00+00:00", periods=5, freq="1h", tz="UTC"
    ).union(pd.date_range("2024-01-01T07:00:00+00:00", periods=5, freq="1h", tz="UTC"))
    df = pd.DataFrame(
        {
            "open": range(len(weekday_stamps)),
            "high": range(len(weekday_stamps)),
            "low": range(len(weekday_stamps)),
            "close": range(len(weekday_stamps)),
            "volume": range(len(weekday_stamps)),
        },
        index=weekday_stamps,
    )
    df.index.name = "date"
    csv_file = write_csv(df)

    provider = CSVOHLCVProvider(
        file=csv_file,
        symbol=symbol_str,
        timeframe=timeframe_str,
    )
    trading_start = weekday_stamps[7]
    provider.set_dates(trading_start.isoformat(), weekday_stamps[-1].isoformat())
    provider.set_required_lookbacks({timeframe_str: 5})

    fetched = provider.fetch_ohlcv(symbol_obj, timeframe_str)

    warmup = fetched[fetched.index < trading_start]
    assert len(warmup) == 5


def test_lookback_on_a_gapless_file_matches_the_calendar_span(
    write_csv, sample_df, symbol_str, symbol_obj, timeframe_str
):
    csv_file = write_csv(sample_df)
    provider = CSVOHLCVProvider(
        file=csv_file,
        symbol=symbol_str,
        timeframe=timeframe_str,
    )
    trading_start = sample_df.index[6]
    provider.set_dates(trading_start.isoformat(), sample_df.index[-1].isoformat())
    provider.set_required_lookbacks({timeframe_str: 3})

    fetched = provider.fetch_ohlcv(symbol_obj, timeframe_str)

    warmup = fetched[fetched.index < trading_start]
    assert len(warmup) == 3
    assert warmup.index[0] == sample_df.index[3]


def test_lookback_reaching_before_the_file_starts_takes_what_the_file_has(
    write_csv, sample_df, symbol_str, symbol_obj, timeframe_str
):
    csv_file = write_csv(sample_df)
    provider = CSVOHLCVProvider(
        file=csv_file,
        symbol=symbol_str,
        timeframe=timeframe_str,
    )
    trading_start = sample_df.index[2]
    provider.set_dates(trading_start.isoformat(), sample_df.index[-1].isoformat())
    provider.set_required_lookbacks({timeframe_str: 10})

    fetched = provider.fetch_ohlcv(symbol_obj, timeframe_str)

    assert fetched.index[0] == sample_df.index[0]


def test_lookback_at_the_files_own_first_row_reaches_no_further(
    write_csv, sample_df, symbol_str, symbol_obj, timeframe_str
):
    csv_file = write_csv(sample_df)
    provider = CSVOHLCVProvider(
        file=csv_file,
        symbol=symbol_str,
        timeframe=timeframe_str,
    )
    trading_start = sample_df.index[0]
    provider.set_dates(trading_start.isoformat(), sample_df.index[-1].isoformat())
    provider.set_required_lookbacks({timeframe_str: 5})

    fetched = provider.fetch_ohlcv(symbol_obj, timeframe_str)

    assert fetched.index[0] == sample_df.index[0]
