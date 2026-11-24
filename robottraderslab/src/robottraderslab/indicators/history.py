import numpy as np
import numpy.typing as npt


def value_periods_ago(
    series: npt.NDArray[np.float64], periods: int
) -> npt.NDArray[np.float64]:
    """The value a series held `periods` candles ago, on every candle.

    Entry i of the result is entry i - periods of the series, so a comparison
    of the two reads a candle against an earlier one, as a crossing does. The
    first `periods` candles have no candle that far back and read NaN, and
    every comparison with NaN is False, so they never signal. A stamp the
    market left empty is NaN in the series, and so is the candle that reads it.

    Args:
        series: One entry per candle, such as a column of closes or the output
            of `sma`.
        periods: How many candles back to read, at least 1.

    Returns:
        A new array of the same length as `series`.

    Raises:
        ValueError: If `periods` is below 1.
    """
    if periods < 1:
        raise ValueError(f"periods must be at least 1, got {periods}")
    past = np.full(series.shape, np.nan)
    past[periods:] = series[:-periods]
    return past
