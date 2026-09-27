import numpy as np
import numpy.typing as npt

from robottraderslab._core import StrategyCriticalError


def published_candles(*series: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    published = ~np.isnan(series[0])
    for values in series[1:]:
        published &= ~np.isnan(values)
    return published


def require_full_window(
    indicator: str, period: int, published: npt.NDArray[np.bool_]
) -> None:
    """A series holding fewer published candles than the window needs carries no
    value on any candle, which a run would spend trading nothing.
    """
    count = int(published.sum())
    if count < period:
        raise StrategyCriticalError(
            f"{indicator}({period}) needs {period} published candles; the series "
            f"carries {count} across {published.size} stamps, "
            f"{published.size - count} of them holding no candle the market "
            f"published. Shorten the period or start the run earlier."
        )


def without_gaps(
    values: npt.NDArray[np.float64], published: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """A market that published every stamp is read where it lies, so a venue
    that never closes spends nothing on the compaction a gap needs.
    """
    return values if published.all() else values[published]


def aligned_to_candles(
    values: npt.NDArray[np.float64], published: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """A computation over the candles `published` selects is as long as the
    stamp axis only when it spanned every stamp of it, which leaves it aligned.
    """
    if values.size == published.size:
        return values

    aligned = np.full(published.size, np.nan, dtype=np.float64)
    aligned[published] = values
    return aligned
