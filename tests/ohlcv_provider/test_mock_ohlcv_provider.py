import numpy as np
import pandas as pd
import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab._core import OhlcvValidationError
from robottraderslab.ohlcv_provider.mock_ohlcv_provider import MockOHLCVProvider


@pytest.fixture
def btc_symbol() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


def _assert_ohlcv_schema(df: pd.DataFrame) -> None:
    expected_cols = {"open", "high", "low", "close", "volume"}
    assert set(df.columns) == expected_cols
    assert isinstance(df.index, pd.DatetimeIndex)
    # Index normalization guarantees tz-aware UTC and monotonicity
    assert df.index.tz is not None
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates


@pytest.mark.parametrize(
    "timeframe",
    ["1m", "1h", "1d", "1w", "1M"],
    ids=["1m", "1h", "1d", "1w", "1M"],
)  # type: ignore[arg-type]
def test_fetch_ohlcv_lookback_mode_generates_expected_length_and_schema(
    btc_symbol: Symbol, timeframe: TimeFrame
) -> None:
    np.random.seed(42)
    lookback = 20
    provider = MockOHLCVProvider(
        lookback=lookback,
        initial_price=10_000,
        volatility=0.01,
    )

    df = provider.fetch_ohlcv(btc_symbol, timeframe)

    assert len(df) == lookback
    _assert_ohlcv_schema(df)
    # OHLC constraints
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["volume"] >= 0).all()


def test_fetch_ohlcv_date_window_generates_inclusive_range_and_schema(
    btc_symbol: Symbol,
) -> None:
    np.random.seed(123)
    timeframe: TimeFrame = "1d"
    start_date = "2024-03-01"
    end_date = "2024-03-10"

    provider = MockOHLCVProvider(
        initial_price=20_000,
        volatility=0.02,
    )
    provider.set_dates(start_date, end_date)

    df = provider.fetch_ohlcv(btc_symbol, timeframe)

    _assert_ohlcv_schema(df)

    expected_index = pd.date_range(start=start_date, end=end_date, freq="1D", tz="UTC")
    assert len(df.index) == len(expected_index)
    assert df.index[0] >= expected_index[0]
    assert df.index[-1] <= expected_index[-1]


def test_fetch_ohlcv_accepts_any_symbol_without_validation_error(
    btc_symbol: Symbol,
) -> None:
    """Test that the provider accepts any symbol without throwing validation errors."""
    np.random.seed(7)
    provider = MockOHLCVProvider(lookback=5)

    df1 = provider.fetch_ohlcv(btc_symbol, "1h")
    df2 = provider.fetch_ohlcv(Symbol.create("ETH/USDT:USDT"), "1h")
    df3 = provider.fetch_ohlcv(Symbol.create("SOL/USDT:USDT"), "1h")

    assert len(df1) == 5
    assert len(df2) == 5
    assert len(df3) == 5
    _assert_ohlcv_schema(df1)
    _assert_ohlcv_schema(df2)
    _assert_ohlcv_schema(df3)


def test_fetch_ohlcv_accepts_any_timeframe_without_validation_error(
    btc_symbol: Symbol,
) -> None:
    """Test that the provider accepts any timeframe without throwing validation errors."""
    np.random.seed(8)
    provider = MockOHLCVProvider(lookback=3)

    df1 = provider.fetch_ohlcv(btc_symbol, "1m")
    df2 = provider.fetch_ohlcv(btc_symbol, "1h")
    df3 = provider.fetch_ohlcv(btc_symbol, "4h")
    df4 = provider.fetch_ohlcv(btc_symbol, "1d")

    for df in [df1, df2, df3, df4]:
        assert len(df) == 3
        _assert_ohlcv_schema(df)


def test_fetch_ohlcv_generates_data_for_multiple_symbol_timeframe_combinations() -> (
    None
):
    """Test that the provider can generate data for multiple symbol/timeframe combinations in sequence."""
    np.random.seed(9)
    provider = MockOHLCVProvider(lookback=2)

    combinations = [
        (Symbol.create("BTC/USDT:USDT"), "1h"),
        (Symbol.create("ETH/USDT:USDT"), "4h"),
        (Symbol.create("SOL/USDT:USDT"), "1d"),
    ]

    results = []
    for symbol, timeframe in combinations:
        df = provider.fetch_ohlcv(symbol, timeframe)
        results.append(df)

    for df in results:
        assert len(df) == 2
        _assert_ohlcv_schema(df)


def test_fetch_ohlcv_lookback_conflict_with_dates_raises(btc_symbol: Symbol) -> None:
    provider = MockOHLCVProvider(lookback=10)
    provider.set_dates("2024-03-01", "2024-03-10")

    with pytest.raises(
        OhlcvValidationError, match="use either lookback or start/end dates"
    ):
        provider.fetch_ohlcv(btc_symbol, "1h")


def test_fetch_ohlcv_invalid_lookback_raises(btc_symbol: Symbol) -> None:
    provider = MockOHLCVProvider(lookback=0)

    with pytest.raises(OhlcvValidationError, match="expected a positive integer"):
        provider.fetch_ohlcv(btc_symbol, "1h")


def test_fetch_ohlcv_lookback_one_step_executes_single_step_path(
    btc_symbol: Symbol,
) -> None:
    np.random.seed(99)
    provider = MockOHLCVProvider(lookback=1)
    df = provider.fetch_ohlcv(btc_symbol, "1d")
    assert len(df) == 1
    _assert_ohlcv_schema(df)


def test_fetch_ohlcv_invalid_timeframe_raises() -> None:
    # Intentionally bypass typing to cover invalid timeframe branch
    bad_tf = "2x"  # not a valid timeframe unit
    sym = Symbol.create("BTC/USDT:USDT")
    provider = MockOHLCVProvider(lookback=5)
    with pytest.raises(OhlcvValidationError, match="Invalid timeframe format"):
        provider.fetch_ohlcv(sym, bad_tf)


def test_fetch_ohlcv_monthly_timeframe_generates_30_day_intervals(
    btc_symbol: Symbol,
) -> None:
    np.random.seed(101)
    tf: TimeFrame = "1M"
    provider = MockOHLCVProvider()
    provider.set_dates("2024-01-15", "2024-04-20")
    df = provider.fetch_ohlcv(btc_symbol, tf)
    _assert_ohlcv_schema(df)
    assert len(df) > 0
    if len(df) > 1:
        # Check that intervals are approximately 30 days
        time_diffs = df.index[1:] - df.index[:-1]
        expected_delta = pd.Timedelta(days=30)
        assert all(diff == expected_delta for diff in time_diffs)


def test_generate_mock_ohlcv_raises_type_error_for_non_datetime_index() -> None:
    """Test that _generate_mock_ohlcv raises TypeError for non-DatetimeIndex input."""
    from robottraderslab.ohlcv_provider.mock_ohlcv_provider import _generate_mock_ohlcv

    non_datetime_index = pd.Index([1, 2, 3])  # Regular Index, not DatetimeIndex

    with pytest.raises(TypeError, match="index must be a pandas.DatetimeIndex"):
        _generate_mock_ohlcv(index=non_datetime_index)


def test_generate_mock_ohlcv_raises_value_error_for_empty_index() -> None:
    """Test that _generate_mock_ohlcv raises ValueError for empty DatetimeIndex."""
    from robottraderslab.ohlcv_provider.mock_ohlcv_provider import _generate_mock_ohlcv

    empty_index = pd.DatetimeIndex([])

    with pytest.raises(ValueError, match="index must contain at least one timestamp"):
        _generate_mock_ohlcv(index=empty_index)


def test_set_strategy_lookback_ignored_when_dates_configured() -> None:
    """Test that set_strategy_lookback is ignored when dates are configured."""
    provider = MockOHLCVProvider()
    provider.set_dates("2024-01-01", "2024-01-02")
    provider.set_strategy_lookback(50)
    assert provider._lookback is None


def test_set_strategy_lookback_updates_when_no_config() -> None:
    """Test that set_strategy_lookback updates lookback when no config lookback."""
    provider = MockOHLCVProvider()
    provider.set_strategy_lookback(50)
    assert provider._lookback == 50


def test_set_strategy_lookback_takes_max_with_config() -> None:
    """Test that set_strategy_lookback takes max of config and strategy lookback."""
    provider = MockOHLCVProvider(lookback=100)
    provider.set_strategy_lookback(50)
    assert provider._lookback == 100

    provider2 = MockOHLCVProvider(lookback=20)
    provider2.set_strategy_lookback(50)
    assert provider2._lookback == 50


def test_fetch_ohlcv_with_no_dates_or_lookback(btc_symbol):
    np.random.seed(200)
    provider = MockOHLCVProvider()

    ohlcv = provider.fetch_ohlcv(btc_symbol, "1d")

    assert not ohlcv.empty
    _assert_ohlcv_schema(ohlcv)


def test_fetch_ohlcv_with_non_numeric_timeframe_prefix(btc_symbol):
    provider = MockOHLCVProvider(lookback=5)

    with pytest.raises(OhlcvValidationError, match="Invalid timeframe format"):
        provider.fetch_ohlcv(btc_symbol, "Xm")
