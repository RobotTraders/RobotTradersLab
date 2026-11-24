import numpy as np
import numpy.typing as npt

from .history import value_periods_ago


def crossover(
    series: npt.NDArray[np.float64], other: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """True on each candle where `series` crosses above `other`.

    `series` is above `other` on the candle and was at or below it on the
    candle before. A candle where either value, or either value of the candle
    before, is NaN reads False.

    Args:
        series: One entry per candle, such as a fast moving average.
        other: One entry per candle, of the same length, such as a slow
            moving average.

    Returns:
        An array of booleans of the same length.
    """
    return (series > other) & (
        value_periods_ago(series, 1) <= value_periods_ago(other, 1)
    )


def crossunder(
    series: npt.NDArray[np.float64], other: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """True on each candle where `series` crosses below `other`.

    `series` is below `other` on the candle and was at or above it on the
    candle before. A candle where either value, or either value of the candle
    before, is NaN reads False.

    Args:
        series: One entry per candle, such as a fast moving average.
        other: One entry per candle, of the same length, such as a slow
            moving average.

    Returns:
        An array of booleans of the same length.
    """
    return (series < other) & (
        value_periods_ago(series, 1) >= value_periods_ago(other, 1)
    )
