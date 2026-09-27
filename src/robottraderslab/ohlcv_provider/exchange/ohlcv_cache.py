from datetime import datetime

import pandas as pd

from robottraderslab._core import Symbol, TimeFrame

_OHLCV_CACHE: dict[
    str, dict[str, dict[str, tuple[datetime, datetime, pd.DataFrame]]]
] = {}


def get_cached_ohlcv(
    exchange: str,
    symbol: Symbol,
    timeframe: TimeFrame,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame | None:
    """Get data from cache if it exists and its dates match the request exactly."""
    exchange_cache = _OHLCV_CACHE.get(exchange, {})
    timeframe_cache = exchange_cache.get(str(timeframe), {})
    symbol_cache = timeframe_cache.get(str(symbol))

    if symbol_cache is None:
        return None

    cached_start, cached_end, cached_df = symbol_cache

    if cached_start == start_date and cached_end == end_date:
        return cached_df

    return None


def store_ohlcv_in_cache(
    exchange: str,
    symbol: Symbol,
    timeframe: TimeFrame,
    start_date: datetime,
    end_date: datetime,
    data: pd.DataFrame,
) -> None:
    """Store data in cache."""
    if exchange not in _OHLCV_CACHE:
        _OHLCV_CACHE[exchange] = {}
    if str(timeframe) not in _OHLCV_CACHE[exchange]:
        _OHLCV_CACHE[exchange][str(timeframe)] = {}

    _OHLCV_CACHE[exchange][str(timeframe)][str(symbol)] = (
        start_date,
        end_date,
        data,
    )


def get_all_cached_ohlcv_for_symbol(exchange: str, symbol: Symbol) -> pd.DataFrame:
    """Get all cached OHLCV for a symbol merged across all timeframes.

    Returns:
        Sorted by timestamp, de-duplicated across timeframes.
    """
    symbol_str = str(symbol)
    all_dataframes = [
        timeframe_cache[symbol_str][2]
        for timeframe_cache in _OHLCV_CACHE.get(exchange, {}).values()
        if symbol_str in timeframe_cache
    ]

    if not all_dataframes:
        return pd.DataFrame()

    combined = pd.concat(all_dataframes)
    combined = combined.sort_index()
    return combined[~combined.index.duplicated(keep="first")]
