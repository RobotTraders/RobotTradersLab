from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from robottraderslab.indicators import MAType, moving_average, trix


@dataclass(frozen=True, slots=True)
class ImpulseSignals:
    """When a side may be entered and when it is closed, one boolean per candle.

    An entry is raised on the first candle of a run of agreeing candles. An
    exit is raised on every candle its side is closed on, a candle the
    indicators cannot decide excepted, so a position still held on a closed
    side leaves on every cycle that books its profile.
    """

    long_entry: npt.NDArray[np.bool_]
    long_exit: npt.NDArray[np.bool_]
    short_entry: npt.NDArray[np.bool_]
    short_exit: npt.NDArray[np.bool_]


def compute_signals(
    close: npt.NDArray[np.float64],
    trix_length: int,
    signal_length: int,
    signal_type: MAType,
    trend_length: int,
    trend_type: MAType,
) -> ImpulseSignals:
    """Return the entry candles and the exit state of both sides.

    A side is open while the TRIX histogram agrees with price sitting the
    same side of the trend average. A run of agreeing candles enters once, on
    its first candle, and the side is closed on every candle outside a run that
    the indicators can decide; a candle the averages cannot yet decide raises
    no exit. The trend length counts the candles `close` publishes, a NaN
    holding none, and the signal length counts the TRIX readings, one fewer
    than those candles; the TRIX length is a span in candles of `close` that
    fills no window. `close` holds at least one candle.

    Raises:
        ValueError: If a windowed trend or signal average finds fewer published
            values than its length.
    """
    indicator = trix(
        close,
        trix_length,
        signal_length,
        signal_type=signal_type,
    )
    trend = moving_average(close, trend_length, trend_type)

    long_open = (indicator.histogram > 0) & (close > trend)
    short_open = (indicator.histogram < 0) & (close < trend)
    decided = ~np.isnan(indicator.histogram) & ~np.isnan(trend)

    return ImpulseSignals(
        long_entry=_starts(long_open),
        long_exit=decided & ~long_open,
        short_entry=_starts(short_open),
        short_exit=decided & ~short_open,
    )


def _starts(open_bars: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    return open_bars & ~_previous(open_bars)


def _previous(open_bars: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    shifted = np.roll(open_bars, 1)
    shifted[0] = False
    return shifted
