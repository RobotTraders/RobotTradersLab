from datetime import datetime, tzinfo
from typing import Literal

import pandas as pd

from .settings import (
    DATE_FORMAT,
    DEFAULT_TIMEZONE,
    USE_TIMEZONE_AWARE,
)

#####################
# Settings accessors
#####################


def get_date_format() -> str:
    """Return the standard date format string for file operations.

    Returns:
        Date strings in this format represent UTC time; parsing should be
        followed by normalization with `to_datetime` when entering the
        provider layer.
    """
    return DATE_FORMAT


def get_default_timezone() -> tzinfo:
    """Return the default timezone used by the provider (UTC)."""
    return DEFAULT_TIMEZONE


def is_timezone_aware() -> bool:
    """Return whether the provider operates in timezone-aware (UTC) mode.

    Notes:
        Internal DataFrames and indices are treated as UTC even when naive
        mode is configured; callers should not rely on naive semantics.
    """
    return USE_TIMEZONE_AWARE


#############################
# Section B: Standardization
##############################


def to_datetime(date_input: str | datetime | None) -> datetime:
    """Convert input into a normalized datetime in the configured timezone.

    Args:
        date_input: A string (tz-aware or naive), a datetime, or None.

    Returns:
        Datetime normalized according to settings:
        - If USE_TIMEZONE_AWARE is True: timezone-aware datetime in DEFAULT_TIMEZONE
        - Otherwise: naive datetime representing DEFAULT_TIMEZONE wall time

        Suitable for inclusive slicing against the UTC-aware DateTimeIndex
        used internally.
    """
    if date_input is None:
        return (
            datetime.now(DEFAULT_TIMEZONE)
            if USE_TIMEZONE_AWARE
            else datetime.now(DEFAULT_TIMEZONE).replace(tzinfo=None)
        )

    if isinstance(date_input, datetime):
        dt = date_input
    elif isinstance(date_input, str):
        # Fast path for common ISO-8601 forms; fallback to pandas for flexibility
        s = date_input
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            ts = pd.to_datetime(s, utc=False)
            dt = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts

    if USE_TIMEZONE_AWARE:
        if dt.tzinfo is None:
            return DEFAULT_TIMEZONE.localize(dt)
        return dt.astimezone(DEFAULT_TIMEZONE)

    if dt.tzinfo is not None:
        return dt.astimezone(DEFAULT_TIMEZONE).replace(tzinfo=None)

    return dt


type TimestampUnit = Literal["D", "s", "ms", "us", "ns"]


def standardize_ohlcv_index_from_timestamp(
    df: pd.DataFrame, timestamp_column: str = "timestamp", unit: TimestampUnit = "ms"
) -> pd.DataFrame:
    """Set a UTC DateTimeIndex from a timestamp column.

    Args:
        df: DataFrame containing a timestamp column.
        timestamp_column: Column name holding timestamps.
        unit: Pandas unit for conversion (e.g., "ms", "s", "us").

    Returns:
        DataFrame with a UTC-aware, de-duplicated, ascending DateTimeIndex.
    """
    if df.empty:
        return df
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], unit=unit, utc=True)
    df = df.set_index(timestamp_column)
    return standardize_ohlcv_index(df)


def standardize_ohlcv_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame's DateTimeIndex to provider standards.

    Args:
        df: Input DataFrame whose index represents timestamps.

    Returns:
        DataFrame with a UTC-normalized, ascending, de-duplicated DateTimeIndex.
    """
    if df.empty:
        return df

    idx = df.index

    if USE_TIMEZONE_AWARE:
        if isinstance(idx, pd.DatetimeIndex):
            tzinfo = idx.tz
            if tzinfo is None:
                idx = idx.tz_localize(DEFAULT_TIMEZONE)
            elif tzinfo != DEFAULT_TIMEZONE:
                idx = idx.tz_convert(DEFAULT_TIMEZONE)
        else:
            idx = pd.to_datetime(idx, utc=True)
    else:
        if isinstance(idx, pd.DatetimeIndex):
            if idx.tz is not None:
                idx = idx.tz_convert(DEFAULT_TIMEZONE).tz_localize(None)
        else:
            idx = pd.to_datetime(idx)

    df.index = idx

    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    return df


############
# Windowing
############


def filter_date_range(
    df: pd.DataFrame,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """Filter a DataFrame by an inclusive datetime window.

    Args:
        df: DataFrame with a normalized UTC-aware DateTimeIndex.
        start_date: Start bound, or None for no lower bound.
        end_date: End bound, or None for no upper bound.

    Returns:
        Input DataFrame if no bounds are provided or it's empty; otherwise an
        inclusive slice using label-based indexing.
    """
    if df.empty or (start_date is None and end_date is None):
        return df

    left = start_date if start_date is not None else None
    right = end_date if end_date is not None else None
    return df.loc[slice(left, right)]
