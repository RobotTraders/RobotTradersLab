import numpy as np
import pandas as pd

from .timeframes import TIMEFRAMES, TimeFrame, to_milliseconds, to_seconds

type CandleStep = pd.Timedelta | pd.offsets.BaseOffset

_NANOSECONDS_PER_MILLISECOND = 1_000_000
_ONE_NANOSECOND = pd.Timedelta(1, "ns")


def candle_closes_after(timeframe: TimeFrame, stamp_ms: int) -> int:
    """When the candle stamped at a moment closes, in UTC milliseconds.

    A month is read off the calendar, since its length depends on which month
    it is; every other timeframe closes a fixed span after its stamp.
    """
    calendar_step = _CALENDAR_STEPS.get(timeframe)
    if calendar_step is None:
        return stamp_ms + to_milliseconds(timeframe)
    closes_at = pd.Timestamp(stamp_ms, unit="ms") + calendar_step
    return int(closes_at.value // _NANOSECONDS_PER_MILLISECOND)


def candles_back(timeframe: TimeFrame, count: int) -> CandleStep:
    """How far back a number of candles reaches from any moment.

    A month spans whichever days it holds, so a count of months answers for
    one, and a multiple of its length for every other timeframe. The moment
    reached back from is any moment, where `candle_step` walks from a candle's
    stamp to the next.
    """
    if timeframe in _CALENDAR_STEPS:
        return pd.DateOffset(months=count)
    return count * pd.Timedelta(seconds=to_seconds(timeframe))


def candle_step(timeframe: TimeFrame) -> CandleStep:
    """The offset from one candle's stamp to the stamp of the next.

    A month runs twenty-eight to thirty-one days, so its step walks the
    calendar; every other timeframe has candles of a single length.
    """
    return _CANDLE_STEPS[timeframe]


def candle_stamps(timeframe: TimeFrame, first_ms: int, last_ms: int) -> np.ndarray:
    """Stamps of every candle the timeframe places between two of its stamps.

    Args:
        first_ms: Stamp the run opens on, in UTC milliseconds.
        last_ms: Stamp the run closes on, included.

    Returns:
        Ascending UTC milliseconds, one entry per candle.
    """
    calendar_step = _CALENDAR_STEPS.get(timeframe)
    if calendar_step is not None:
        return _stamps_along_the_calendar(calendar_step, first_ms, last_ms)
    span_ms = to_milliseconds(timeframe)
    return np.arange(first_ms, last_ms + span_ms, span_ms, dtype=np.int64)


def last_per_month(series: pd.Series) -> pd.Series:
    """The last value of every calendar month of a series stamped at candle
    closes, each labelled with its month's first day.

    A moment on the first midnight of a month closes the candle before it, so
    its value ends the month before: each value is placed at the instant
    before its moment, which lies inside the candle it closes.
    """
    inside_the_closed_candle = pd.DatetimeIndex(series.index) - _ONE_NANOSECOND
    return series.set_axis(inside_the_closed_candle).resample("MS").last()


def _stamps_along_the_calendar(
    step: pd.offsets.BaseOffset, first_ms: int, last_ms: int
) -> np.ndarray:
    stamps = pd.date_range(
        pd.Timestamp(first_ms, unit="ms"), pd.Timestamp(last_ms, unit="ms"), freq=step
    )
    return stamps.to_numpy(dtype="datetime64[ms]").astype(np.int64)


_CALENDAR_STEPS: dict[TimeFrame, pd.offsets.BaseOffset] = {
    "1M": pd.offsets.MonthBegin(1)
}

_CANDLE_STEPS: dict[TimeFrame, CandleStep] = {
    timeframe: pd.Timedelta(seconds=to_seconds(timeframe)) for timeframe in TIMEFRAMES
}
_CANDLE_STEPS.update(_CALENDAR_STEPS)
