import logging

import pandas as pd

from robottraderslab._core import (
    TIMEFRAMES,
    OHLCVProviderProtocol,
    Symbol,
    TimeFrame,
    to_seconds,
)
from robottraderslab.exceptions import ExchangeRecoverableError

logger = logging.getLogger(__name__)

_FALLBACK_TIMEFRAME: TimeFrame = "1d"


def resolve_reference(
    reference_symbol: str | None,
    reference_timeframe: TimeFrame | None,
    equity_index: pd.Index,
    ohlcv_provider: OHLCVProviderProtocol,
) -> tuple[pd.Series | None, str | None]:
    """The declared text is parsed here, so the venue and the report read the
    reference in the one form.
    """
    if reference_symbol is None:
        return None, None

    symbol = Symbol.create(reference_symbol)
    price = resolve_reference_price(
        symbol, equity_index, ohlcv_provider, preferred_timeframe=reference_timeframe
    )
    return price, str(symbol)


def resolve_reference_price(
    symbol: Symbol,
    equity_index: pd.Index,
    ohlcv_provider: OHLCVProviderProtocol,
    *,
    preferred_timeframe: TimeFrame | None,
) -> pd.Series | None:
    """A report without a benchmark is still a report, so a reference the venue
    cannot serve leaves the run uncompared.

    Args:
        equity_index: The run's timestamps; the candles come at the largest
            timeframe that fits their spacing when none is preferred.
        ohlcv_provider: Source of the candles, read from its cache first.
        preferred_timeframe: Timeframe to fetch at when the cache has nothing.
    """
    cached = ohlcv_provider.get_all_cached_ohlcv_for_symbol(symbol)
    if not cached.empty:
        return cached["close"].copy()

    timeframe = preferred_timeframe or _infer_timeframe_from_equity_index(equity_index)
    try:
        return ohlcv_provider.fetch_ohlcv(symbol, timeframe)["close"].copy()
    except ExchangeRecoverableError:
        logger.warning(
            "Failed to resolve reference price for %s", symbol, exc_info=True
        )
        return None


def _infer_timeframe_from_equity_index(equity_index: pd.Index) -> TimeFrame:
    if len(equity_index) < 2:
        return _FALLBACK_TIMEFRAME
    median_step = pd.Timedelta(equity_index.to_series().diff().median())
    fitting = [tf for tf in TIMEFRAMES if to_seconds(tf) <= median_step.total_seconds()]
    return max(fitting, key=to_seconds, default=min(TIMEFRAMES, key=to_seconds))
