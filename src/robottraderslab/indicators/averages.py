from enum import StrEnum

import numpy as np
import numpy.typing as npt
import pandas as pd

from .gaps import (
    aligned_to_candles,
    published_candles,
    require_full_window,
    without_gaps,
)


class MAType(StrEnum):
    """The kinds of moving average an indicator can be built on."""

    SMA = "SMA"
    EMA = "EMA"
    WMA = "WMA"


def sma(close: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    """Simple moving average of a close-price series, NaN until the window
    is full.

    The window spans `period` candles the market published, so the stamps it
    leaves empty over a weekend or a holiday carry no weight in the average.

    Raises:
        StrategyCriticalError: If the series holds fewer published candles than
            `period`.
    """
    published = published_candles(close)
    require_full_window(MAType.SMA, period, published)

    averaged = (
        pd.Series(without_gaps(close, published)).rolling(period).mean().to_numpy()
    )
    return aligned_to_candles(averaged, published)


def ema(close: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    """Exponential moving average of a close-price series, carrying a value
    from the first candle.

    The average decays from one published candle to the next, so the stamps a
    market leaves empty over a weekend or a holiday age it by nothing.
    """
    published = published_candles(close)

    averaged = (
        pd.Series(without_gaps(close, published))
        .ewm(span=period, adjust=False)
        .mean()
        .to_numpy()
    )
    return aligned_to_candles(averaged, published)


def wma(close: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    """Linearly weighted moving average of a close-price series, NaN until
    the window is full.

    The window spans `period` candles the market published, so the stamps it
    leaves empty over a weekend or a holiday carry no weight in the average.

    Raises:
        StrategyCriticalError: If the series holds fewer published candles than
            `period`.
    """
    weights = np.arange(1, period + 1, dtype=float)
    divisor = period * (period + 1) / 2

    def _weighted(window: np.ndarray) -> float:
        return float(np.dot(window, weights) / divisor)

    published = published_candles(close)
    require_full_window(MAType.WMA, period, published)

    averaged = (
        pd.Series(without_gaps(close, published))
        .rolling(period)
        .apply(_weighted, raw=True)
        .to_numpy()
    )
    return aligned_to_candles(averaged, published)


def moving_average(
    close: npt.NDArray[np.float64], period: int, ma_type: MAType = MAType.SMA
) -> npt.NDArray[np.float64]:
    """Moving average of a close-price series, of the kind `ma_type` names.

    Raises:
        StrategyCriticalError: If a windowed average is asked for and the series
            holds fewer published candles than `period`.
    """
    return _AVERAGES_BY_TYPE[ma_type](close, period)


_AVERAGES_BY_TYPE = {
    MAType.SMA: sma,
    MAType.EMA: ema,
    MAType.WMA: wma,
}
