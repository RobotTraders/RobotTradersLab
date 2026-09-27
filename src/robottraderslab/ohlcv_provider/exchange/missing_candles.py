import logging
from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from robottraderslab._core import Symbol, TimeFrame, candle_stamps

from ..types import OhlcvColumn

logger = logging.getLogger(__name__)

type MarketOpenMask = Callable[[pd.DatetimeIndex], np.ndarray | None]


def handle_missing_candles(
    ohlcv: pd.DataFrame,
    symbol: Symbol,
    timeframe: TimeFrame,
    market_open_mask: MarketOpenMask | None = None,
) -> pd.DataFrame:
    """Detect missing candles in OHLCV data and fill gaps with NaN values.

    Which candles the range ought to hold comes from the timeframe's own
    stamps, since the length of a month depends on which month it is.

    Args:
        market_open_mask: Marks which timestamps fall within market trading
            hours; gaps outside those hours are filled without a warning.
    """
    ts_vals = ohlcv[OhlcvColumn.TIMESTAMP].to_numpy(dtype="int64", copy=False)

    expected = candle_stamps(timeframe, ts_vals[0], ts_vals[-1])
    missing = np.setdiff1d(expected, ts_vals, assume_unique=True)

    if missing.size == 0:
        return ohlcv

    _warn_unexpected_gaps(missing, expected, symbol, market_open_mask)

    missing_ohlcv = pd.DataFrame(
        {
            OhlcvColumn.TIMESTAMP: missing,
            OhlcvColumn.OPEN: np.nan,
            OhlcvColumn.HIGH: np.nan,
            OhlcvColumn.LOW: np.nan,
            OhlcvColumn.CLOSE: np.nan,
            OhlcvColumn.VOLUME: np.nan,
        }
    )

    return pd.concat([ohlcv, missing_ohlcv], ignore_index=True, sort=False)


def trades_between(
    market_open_mask: MarketOpenMask,
    timeframe: TimeFrame,
    start: datetime,
    end: datetime,
) -> bool:
    """A source answering no mask is one whose market never closes."""
    end_ms = _milliseconds(end)
    stamps = candle_stamps(timeframe, _milliseconds(start), end_ms)
    open_hours = market_open_mask(
        pd.to_datetime(stamps[stamps < end_ms], unit="ms", utc=True)
    )
    return open_hours is None or bool(open_hours.any())


def _warn_unexpected_gaps(
    missing: np.ndarray,
    expected: np.ndarray,
    symbol: Symbol,
    market_open_mask: MarketOpenMask | None,
) -> None:
    unexpected = _market_open_subset(missing, market_open_mask)
    if unexpected.size:
        logger.warning(
            f"{symbol}: {unexpected.size} missing candles replaced with NaN's: "
            f"{_format_ranges(unexpected, expected)}"
        )


def _market_open_subset(
    missing: np.ndarray, market_open_mask: MarketOpenMask | None
) -> np.ndarray:
    if market_open_mask is None:
        return missing
    open_mask = market_open_mask(pd.to_datetime(missing, unit="ms", utc=True))
    return missing if open_mask is None else missing[open_mask]


def _format_ranges(timestamps: np.ndarray, expected: np.ndarray) -> str:
    positions = np.searchsorted(expected, timestamps)
    breaks = np.flatnonzero(np.diff(positions) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [timestamps.size - 1]))

    parts = []
    for start, end in zip(starts, ends):
        first = _format_timestamp(timestamps[start])
        if start == end:
            parts.append(first)
        else:
            parts.append(f"{first} -> {_format_timestamp(timestamps[end])}")
    return ", ".join(parts)


def _format_timestamp(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M"
    )


def _milliseconds(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)
