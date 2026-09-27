from datetime import datetime

import numpy as np
import pandas as pd

from robottraderslab._core import (
    OHLCVProviderProtocol,
    OhlcvValidationError,
    Symbol,
    TimeFrame,
)

from .date_utils import (
    get_default_timezone,
    is_timezone_aware,
    standardize_ohlcv_index,
    to_datetime,
)

SECONDS_PER_DAY = 86400


class MockOHLCVProvider(OHLCVProviderProtocol):
    """Mock provider that generates OHLCV data for any symbol/timeframe combination.

    Supports live-like (lookback) and backtest (start_date/end_date) modes.
    """

    def __init__(
        self,
        *,
        lookback: int | None = None,
        initial_price: float = 20000,
        volatility: float = 0.02,
    ) -> None:
        self._lookback = lookback
        self._initial_price = initial_price
        self._volatility = volatility

        self._start_date: datetime | None = None
        self._end_date: datetime | None = None

    def set_strategy_lookback(self, lookback: int) -> None:
        """Set strategy-computed lookback.

        Only applied when using lookback mode (no explicit dates configured).
        Takes the maximum of config lookback and strategy lookback to ensure
        both user preferences and strategy requirements are satisfied.

        Args:
            lookback: Minimum number of candles required by the strategy
        """
        if self._start_date is None and self._end_date is None:
            self._lookback = max(self._lookback or 0, lookback)

    def set_dates(self, start_date: str, end_date: str) -> None:
        """
        Set dates for data loading.

        Args:
            start_date: Start date (ISO format or datetime string)
            end_date: End date (ISO format or datetime string)
        """
        self._start_date = to_datetime(start_date)
        self._end_date = to_datetime(end_date)

    def get_all_cached_ohlcv_for_symbol(self, symbol: Symbol) -> pd.DataFrame:
        """Mock provider has no cache — returns empty DataFrame."""
        return pd.DataFrame()

    def fetch_ohlcv(self, symbol: Symbol, timeframe: TimeFrame) -> pd.DataFrame:
        """Generate mock OHLCV for the requested symbol/timeframe."""
        if self._lookback is not None:
            if self._start_date is not None or self._end_date is not None:
                raise OhlcvValidationError(
                    "Invalid configuration: use either lookback or start/end dates, not both. "
                    f"Received lookback={self._lookback}, start_date={self._start_date}, end_date={self._end_date}."
                )
            if self._lookback <= 0:
                raise OhlcvValidationError(
                    f"Invalid lookback: expected a positive integer, got {self._lookback}."
                )
            index = _build_index_from_lookback(timeframe, self._lookback)
        else:
            index = _build_index_from_dates(timeframe, self._start_date, self._end_date)

        return _generate_mock_ohlcv(
            index=index,
            initial_price=self._initial_price,
            volatility=self._volatility,
        )


def _build_index_from_lookback(timeframe: TimeFrame, lookback: int) -> pd.DatetimeIndex:
    """Build a datetime index for the last N candles aligned to timeframe."""
    freq = _timeframe_to_pandas_freq(timeframe)
    end_dt = _aligned_now(freq)
    idx = pd.date_range(end=end_dt, periods=lookback, freq=freq)
    df = pd.DataFrame(index=idx)
    df = standardize_ohlcv_index(df)
    return pd.DatetimeIndex(df.index)


def _build_index_from_dates(
    timeframe: TimeFrame, start_date: datetime | None, end_date: datetime | None
) -> pd.DatetimeIndex:
    """Build a datetime index between start and end aligned to timeframe."""
    freq = _timeframe_to_pandas_freq(timeframe)
    start_dt, end_dt = _resolve_date_window(start_date, end_date)
    start_dt = _align_to_freq(start_dt, freq)
    end_dt = _align_to_freq(end_dt, freq)
    idx = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    df = pd.DataFrame(index=idx)
    df = standardize_ohlcv_index(df)
    return pd.DatetimeIndex(df.index)


def _resolve_date_window(
    start_date: datetime | None, end_date: datetime | None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve start/end datetimes using shared date utils and defaults."""
    if start_date is not None and end_date is not None:
        return pd.Timestamp(start_date), pd.Timestamp(end_date)

    end_dt = pd.Timestamp(to_datetime(None))
    start_dt = end_dt - pd.Timedelta(days=100)
    return start_dt, end_dt


def _aligned_now(freq: str) -> pd.Timestamp:
    dt = pd.Timestamp(to_datetime(None))
    tz = get_default_timezone()
    if is_timezone_aware():
        dt = dt.tz_convert(tz)
    return _align_to_freq(dt, freq)


def _timeframe_to_pandas_freq(timeframe: TimeFrame) -> str:
    """Convert TimeFrame to pandas frequency, using only fixed frequencies."""
    unit = timeframe[-1]
    try:
        count = int(timeframe[:-1])
    except ValueError as e:
        raise OhlcvValidationError(f"Invalid timeframe format: {timeframe}") from e
    match unit:
        case "m":
            return f"{count}min"
        case "h":
            return f"{count}h"
        case "d":
            return f"{count}D"
        case "w":
            # Use 7-day periods instead of weekly
            return f"{count * 7}D"
        case "M":
            # Use 30-day periods instead of monthly
            return f"{count * 30}D"
        case _:
            raise OhlcvValidationError(f"Invalid timeframe format: {timeframe}")


def _align_to_freq(dt: pd.Timestamp, freq: str) -> pd.Timestamp:
    """Align timestamp to frequency boundary using only pandas' fixed freq support."""
    return dt.floor(freq)


def _generate_mock_ohlcv(
    *,
    index: pd.DatetimeIndex,
    volatility: float = 0.02,
    initial_price: float = 10000,
) -> pd.DataFrame:
    """Generate synthetic OHLCV on a provided UTC datetime index.

    Volatility is scaled per-step by sqrt(step_seconds / seconds_per_day).
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas.DatetimeIndex")
    if len(index) == 0:
        raise ValueError("index must contain at least one timestamp")

    step_seconds = _compute_step_seconds(index)
    per_step_vol = _compute_per_step_volatility(step_seconds, volatility)
    closes = _simulate_closes(initial_price, per_step_vol)
    opens, highs, lows = _generate_ohlc_from_closes(closes, per_step_vol)
    volumes = _generate_volumes(index, initial_price, step_seconds)

    ohlcv_df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=index,
    )

    return ohlcv_df


def _compute_step_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Compute per-step durations in seconds for the given index."""
    if len(index) >= 2:
        diffs = np.diff(index) / np.timedelta64(1, "s")
        step_seconds = np.empty(len(index))
        step_seconds[0] = diffs[0]
        step_seconds[1:] = diffs
    else:
        step_seconds = np.array([SECONDS_PER_DAY], dtype=float)
    return step_seconds


def _compute_per_step_volatility(
    step_seconds: np.ndarray, base_volatility: float
) -> np.ndarray:
    """Scale base volatility by sqrt(step_seconds / day) for each step."""
    return base_volatility * np.sqrt(step_seconds / SECONDS_PER_DAY)


def _simulate_closes(
    initial_price: float, per_step_volatility: np.ndarray
) -> np.ndarray:
    """Simulate close prices using a simple normal-return random walk."""
    random_returns = np.random.normal(
        0.0, per_step_volatility, len(per_step_volatility)
    )
    price_multipliers = np.cumprod(1 + random_returns)
    return initial_price * price_multipliers


def _generate_ohlc_from_closes(
    closes: np.ndarray, per_step_volatility: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive open, high, low from closes using per-step volatility as spread."""
    spread = per_step_volatility / 2
    opens = closes * (1 + np.random.normal(0.0, spread, len(closes)))
    highs = np.maximum(opens, closes) * (
        1 + np.abs(np.random.normal(0.0, spread, len(closes)))
    )
    lows = np.minimum(opens, closes) * (
        1 - np.abs(np.random.normal(0.0, spread, len(closes)))
    )
    return opens, highs, lows


def _generate_volumes(
    index: pd.DatetimeIndex, initial_price: float, step_seconds: np.ndarray
) -> np.ndarray:
    """Generate synthetic volumes inversely proportional to step duration."""
    base_volume = initial_price / 10
    volume_scale = SECONDS_PER_DAY / step_seconds
    return (
        np.random.uniform(low=base_volume, high=initial_price, size=len(index))
        / volume_scale
    )
