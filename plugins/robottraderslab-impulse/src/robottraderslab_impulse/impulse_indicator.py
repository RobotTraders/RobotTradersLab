from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from robottraderslab.indicators import MAType, moving_average, trix


@dataclass(frozen=True, slots=True)
class ImpulseSignals:
    """The candles a side may be entered or exited on, one boolean per candle."""

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
    """Return the entry and exit candles for both sides.

    A side is open while the TRIX histogram agrees with price sitting the
    same side of the trend average, so a run of agreeing candles trades once.
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

    return ImpulseSignals(
        long_entry=_starts(long_open),
        long_exit=_ends(long_open),
        short_entry=_starts(short_open),
        short_exit=_ends(short_open),
    )


def _starts(open_bars: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    return open_bars & ~_previous(open_bars)


def _ends(open_bars: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    return ~open_bars & _previous(open_bars)


def _previous(open_bars: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    shifted = np.roll(open_bars, 1)
    shifted[0] = False
    return shifted
