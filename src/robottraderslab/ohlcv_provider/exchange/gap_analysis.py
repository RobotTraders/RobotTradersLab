from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from robottraderslab._core import CandleStep, TimeFrame, candle_step

from ..date_utils import to_datetime
from ..types import DateRange, DateRanges, OhlcvColumn
from .missing_candles import MarketOpenMask


def compute_storage_gaps(
    existing_data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    timeframe: TimeFrame,
    market_open_mask: MarketOpenMask,
    earliest_available: datetime | None = None,
    known_empty: Sequence[DateRange] = (),
) -> DateRanges:
    """Compute the ranges within [start_date, end_date] the storage holds no
    candle for.

    Assumes timeframe has been validated by the caller and the stored index
    ascending, as the repository contract guarantees. A stored row filled
    where the venue published no candle covers nothing inside market hours,
    so the venue is asked for the candle again; outside them it stands for
    the closure it fills.

    Args:
        market_open_mask: The venue's calendar, which tells a hole the venue
            may still fill from a closure it never will.
        earliest_available: If known, gaps before this timestamp are skipped
            to avoid re-downloading data the exchange does not have.
        known_empty: Ranges the exchange already reported as holding no data,
            so a gap inside one of them is not requested again.
    """
    effective_start = effective_window_start(start_date, earliest_available)
    if effective_start >= end_date:
        return []

    if existing_data.empty:
        return _drop_known_empty([(effective_start, end_date)], known_empty)

    step = candle_step(timeframe)

    window_idx = _covered_stamps(
        existing_data, effective_start, end_date, market_open_mask
    )
    if len(window_idx) == 0:
        return _drop_known_empty([(effective_start, end_date)], known_empty)

    missing: DateRanges = []
    missing.extend(
        _compute_missing_before_first(window_idx, step, effective_start, end_date)
    )
    missing.extend(
        _compute_internal_gap_ranges(window_idx, step, effective_start, end_date)
    )
    missing.extend(
        _compute_missing_after_last(window_idx, step, effective_start, end_date)
    )

    missing = [(to_datetime(start), to_datetime(end)) for start, end in missing]
    return _drop_known_empty(missing, known_empty)


def effective_window_start(
    start_date: datetime, earliest_available: datetime | None
) -> datetime:
    """Start of the part of a window the exchange can fill.

    Candles dated before the exchange's known boundary do not exist to fetch,
    so a window reaching further back starts at that boundary.
    """
    if earliest_available is not None and earliest_available > start_date:
        return earliest_available
    return start_date


def _drop_known_empty(
    missing: DateRanges, known_empty: Sequence[DateRange]
) -> DateRanges:
    return [
        (start, end)
        for start, end in missing
        if not any(
            empty_start <= start and end <= empty_end
            for empty_start, empty_end in known_empty
        )
    ]


def _covered_stamps(
    existing_data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    market_open_mask: MarketOpenMask,
) -> pd.DatetimeIndex:
    idx = existing_data.index
    inside_window = (idx >= start_date) & (idx <= end_date)
    covered = inside_window & ~_holes_inside_market_hours(
        existing_data, inside_window, market_open_mask
    )
    return pd.DatetimeIndex(idx[covered].unique())


def _holes_inside_market_hours(
    existing_data: pd.DataFrame,
    inside_window: np.ndarray,
    market_open_mask: MarketOpenMask,
) -> np.ndarray:
    """The calendar is asked about the window's holes alone, since a store
    holds years of closures a run never reads.
    """
    holes: np.ndarray = (
        existing_data[OhlcvColumn.CLOSE].isna().to_numpy() & inside_window
    )
    if not holes.any():
        return holes
    open_hours = market_open_mask(pd.DatetimeIndex(existing_data.index)[holes])
    if open_hours is None:
        return holes
    inside_hours = np.zeros_like(holes)
    inside_hours[np.flatnonzero(holes)[open_hours]] = True
    return inside_hours


def _compute_missing_before_first(
    window_idx: pd.DatetimeIndex,
    step: CandleStep,
    start_date: datetime,
    end_date: datetime,
) -> DateRanges:
    """Range before the first stored candle, if a whole candle fits in it.

    A window need not start on a candle boundary, so the first candle inside it
    begins after the start. Less room than a whole candle belongs to one that
    runs from before the window, which is not this window's to fetch.
    """
    if len(window_idx) == 0:
        return [(start_date, end_date)]  # pragma: no cover
    first_ts = window_idx[0]
    candidate_ends = start_date + step
    if candidate_ends > first_ts:
        return []
    return [(start_date, first_ts.to_pydatetime())]


def _compute_internal_gap_ranges(
    window_idx: pd.DatetimeIndex,
    step: CandleStep,
    start_date: datetime,
    end_date: datetime,
) -> DateRanges:
    """Ranges between stored candles that hold candles the storage lacks.

    A gap opens where the stamp following a stored candle is not the next one
    stored, and it is requested from that following stamp so the exchange
    recognises the candle being asked for.
    """
    if len(window_idx) < 2:
        return []
    next_stamps = window_idx[:-1] + step
    gap_mask = window_idx[1:] > next_stamps
    if not gap_mask.any():
        return []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    gap_starts_raw = next_stamps[gap_mask]
    gap_starts = gap_starts_raw.where(gap_starts_raw >= start_ts, start_ts)
    gap_ends_raw = window_idx[1:][gap_mask]
    gap_ends = gap_ends_raw.where(gap_ends_raw <= end_ts, end_ts)
    valid = gap_starts < gap_ends
    if not valid.any():
        return []  # pragma: no cover
    return list(zip(gap_starts[valid].to_pydatetime(), gap_ends[valid].to_pydatetime()))


def _compute_missing_after_last(
    window_idx: pd.DatetimeIndex,
    step: CandleStep,
    start_date: datetime,
    end_date: datetime,
) -> DateRanges:
    """Range after the last stored candle, if the window holds a further one.

    A candle is stamped with the start of its period, so the one after the last
    stored begins a step later, and the window is inclusive of a candle stamped
    exactly at its end. Anything short of that stamp holds no candle to fetch.
    The range runs from the last stored stamp, since a source asked from there
    answers with the candles that follow it.
    """
    if len(window_idx) == 0:
        return [(start_date, end_date)]  # pragma: no cover
    last_ts = window_idx[-1]
    if last_ts + step > end_date:
        return []
    return [(last_ts.to_pydatetime(), end_date)]
