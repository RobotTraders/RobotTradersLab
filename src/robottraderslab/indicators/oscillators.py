from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from .averages import MAType, ema, moving_average
from .gaps import aligned_to_candles, published_candles, without_gaps


@dataclass(frozen=True, slots=True)
class Trix:
    """The TRIX oscillator and its signal line, one entry per candle in each
    array.

    Attributes:
        triple_ema: The series `trix` is read from.
        trix: In percent.
        signal: The moving average of `trix` over `signal_length` candles.
        histogram: `trix` minus `signal`.
    """

    triple_ema: npt.NDArray[np.float64]
    trix: npt.NDArray[np.float64]
    signal: npt.NDArray[np.float64]
    histogram: npt.NDArray[np.float64]


def trix(
    close: npt.NDArray[np.float64],
    length: int,
    signal_length: int,
    signal_type: MAType = MAType.EMA,
) -> Trix:
    """Compute the TRIX oscillator and its signal line.

    Triple smoothing is always EMA, by definition of TRIX. The rate of change
    is read from one published candle to the one before it, so a stamp a
    market leaves empty carries no reading.

    Args:
        length: Span of each of the three EMA passes over the close.
        signal_length: Period of the average the signal line is drawn with.
        signal_type: The kind of moving average the signal line is drawn with.

    Raises:
        StrategyCriticalError: If the signal average needs a full window and the
            oscillator holds fewer published values than `signal_length`.
    """
    published = published_candles(close)
    triple_ema = ema(ema(ema(close, length), length), length)
    oscillator = aligned_to_candles(
        np.asarray(
            pd.Series(without_gaps(triple_ema, published)).pct_change() * 100,
            dtype=np.float64,
        ),
        published,
    )
    signal = moving_average(oscillator, signal_length, signal_type)
    histogram = oscillator - signal
    return Trix(
        triple_ema=triple_ema,
        trix=oscillator,
        signal=signal,
        histogram=histogram,
    )
