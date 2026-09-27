import numpy as np
import numpy.typing as npt
import pandas as pd

from .gaps import (
    aligned_to_candles,
    published_candles,
    require_full_window,
    without_gaps,
)

_MIDLINE_NAME = "DCM"


def donchian_midline(
    high: npt.NDArray[np.float64], low: npt.NDArray[np.float64], period: int
) -> npt.NDArray[np.float64]:
    """Midline of the Donchian channel, halfway between the highest high and
    the lowest low of the window, NaN until the window is full.

    The channel spans `period` candles the market published, so the stamps it
    leaves empty over a weekend or a holiday hold neither extreme.

    Raises:
        StrategyCriticalError: If the series holds fewer published candles than
            `period`.
    """
    published = published_candles(high, low)
    require_full_window(_MIDLINE_NAME, period, published)

    highest = pd.Series(without_gaps(high, published)).rolling(period).max()
    lowest = pd.Series(without_gaps(low, published)).rolling(period).min()
    return aligned_to_candles(((highest + lowest) / 2).to_numpy(), published)
