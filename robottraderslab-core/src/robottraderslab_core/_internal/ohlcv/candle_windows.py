from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ..symbol import Symbol
from ..timeframes import TimeFrame
from .ohlcv_requirements import OHLCVRequirements
from .ohlcvs import OHLCVs


@dataclass(frozen=True, slots=True)
class CandleWindow:
    """The span of one candle, from its open up to but excluding its close."""

    start: datetime
    end: datetime

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def contains(self, timestamp: datetime) -> bool:
        """Whether an event at the timestamp belongs to this candle."""
        return self.start <= timestamp < self.end


type CandleWindows = dict[Symbol, CandleWindow]


def candle_windows(
    requirements: OHLCVRequirements,
    ohlcvs: OHLCVs,
    timeframes: set[TimeFrame],
) -> CandleWindows:
    """Return the candle that just closed for each symbol of the timeframes.

    Timestamps come back as UTC, the zone exchanges report their fills in. A
    symbol declared on several of the timeframes takes the shortest one, whose
    candle attributes an event without ambiguity.

    Args:
        requirements: OHLCV declarations, naming which symbols each timeframe
            covers.
        ohlcvs: Loaded market data, holding the newest candle per timeframe.
        timeframes: Timeframes whose candle just closed.

    Returns:
        The closed candle per symbol, for the declared symbols of those
        timeframes.
    """
    closed = {timeframe: _window_of(ohlcvs, timeframe) for timeframe in timeframes}
    windows: CandleWindows = {}
    for requirement in requirements._get_all():
        window = closed.get(requirement.timeframe)
        if window is None:
            continue
        booked = windows.get(requirement.symbol)
        if booked is None or window.duration < booked.duration:
            windows[requirement.symbol] = window
    return windows


def _window_of(ohlcvs: OHLCVs, timeframe: TimeFrame) -> CandleWindow:
    return CandleWindow(
        start=_as_utc(ohlcvs._last_candle_open(timeframe)),
        end=_as_utc(ohlcvs._last_candle_close(timeframe)),
    )


def _as_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)
