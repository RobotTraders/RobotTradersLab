from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Literal, get_args

TimeFrame = Literal[
    "1m",
    "2m",
    "3m",
    "5m",
    "15m",
    "30m",
    "45m",
    "1h",
    "2h",
    "3h",
    "4h",
    "1d",
    "1w",
    "1M",
]

TIMEFRAMES: tuple[TimeFrame, ...] = get_args(TimeFrame)

_SECONDS_PER_MINUTE = 60
_MILLISECONDS_PER_SECOND = 1000
_ONE_MICROSECOND = timedelta(microseconds=1)
_ONE_DAY = timedelta(days=1)
_ONE_WEEK = timedelta(days=7)


def to_seconds(tf: TimeFrame) -> int:
    """How long one candle lasts, a month counted as thirty days.

    The monthly answer is nominal, so only the ordering of timeframes and the
    sizing of a request window may rest on it. Placing a candle takes a step or
    a span, which read a month's length off the calendar.
    """
    match tf[-1]:
        case "m":  # minutes
            return int(tf[:-1]) * 60
        case "h":  # hours
            return int(tf[:-1]) * 60 * 60
        case "d":  # days
            return int(tf[:-1]) * 60 * 60 * 24
        case "w":  # weeks of 7 days
            return int(tf[:-1]) * 60 * 60 * 24 * 7
        case "M":  # months of 30 days
            return int(tf[:-1]) * 60 * 60 * 24 * 30
        case _:  # pragma: no cover
            raise ValueError(f"Invalid timeframe {tf}")  # pragma: no cover


def to_milliseconds(tf: TimeFrame) -> int:
    """How long one candle lasts in the unit venues stamp candles with.

    The monthly answer is nominal, so only the sizing of a request window
    may rest on it.
    """
    return to_seconds(tf) * _MILLISECONDS_PER_SECOND


def candle_closes_at(timeframe: TimeFrame, moment: datetime) -> bool:
    """Whether a candle of the timeframe closes in the minute of the moment.

    Weekly candles close on Monday midnight and monthly candles on the first
    of the month, following exchange convention.

    Args:
        moment: Aware UTC datetime.
    """
    return _BOUNDARY_CHECKS[timeframe](moment)


def newest_settled_candle_start(timeframe: TimeFrame, moment: datetime) -> datetime:
    """Stamp of the newest candle whose period has fully elapsed at an aware moment.

    The candle holding the moment is still forming, so the answer is the one
    before it. Weeks run from Monday midnight and months from the first, as the
    exchanges stamp them.
    """
    match timeframe:
        case "1w":
            return _week_start(moment) - _ONE_WEEK
        case "1M":
            return _month_start(_month_start(moment) - _ONE_DAY)
        case _:
            period = to_seconds(timeframe)
            settled_periods = int(moment.timestamp()) // period - 1
            return datetime.fromtimestamp(settled_periods * period, tz=timezone.utc)


def candle_stamp_before(timeframe: TimeFrame, moment: datetime) -> datetime:
    """Stamp of the newest candle opening strictly before an aware moment.

    A moment on the grid opens a candle of its own, so the answer is the one
    before it; a moment inside a candle answers that candle's stamp. Weeks run
    from Monday midnight and months from the first, as the exchanges stamp
    them.
    """
    just_before = moment - _ONE_MICROSECOND
    match timeframe:
        case "1w":
            return _week_start(just_before)
        case "1M":
            return _month_start(just_before)
        case _:
            period = to_seconds(timeframe)
            opened_periods = int(just_before.timestamp()) // period
            return datetime.fromtimestamp(opened_periods * period, tz=timezone.utc)


def _epoch_modulo_check(timeframe: TimeFrame) -> Callable[[datetime], bool]:
    seconds = to_seconds(timeframe)
    return lambda moment: int(moment.timestamp()) % seconds < _SECONDS_PER_MINUTE


def _closes_monday_midnight(moment: datetime) -> bool:
    return moment.weekday() == 0 and moment.hour == 0 and moment.minute == 0


def _closes_first_of_month(moment: datetime) -> bool:
    return moment.day == 1 and moment.hour == 0 and moment.minute == 0


_BOUNDARY_CHECKS: dict[TimeFrame, Callable[[datetime], bool]] = {
    timeframe: _epoch_modulo_check(timeframe) for timeframe in TIMEFRAMES
}
_BOUNDARY_CHECKS["1w"] = _closes_monday_midnight
_BOUNDARY_CHECKS["1M"] = _closes_first_of_month


def _week_start(moment: datetime) -> datetime:
    return _midnight(moment) - timedelta(days=moment.weekday())


def _month_start(moment: datetime) -> datetime:
    return _midnight(moment).replace(day=1)


def _midnight(moment: datetime) -> datetime:
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)
