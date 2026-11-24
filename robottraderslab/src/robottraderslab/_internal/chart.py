from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

type ChartPane = Literal["price", "separate"]
type ChartShape = Literal["line", "histogram"]


class DrawdownUnit(StrEnum):
    """What a drawdown's values are counted in, so an axis can label them."""

    PERCENT = "percent"
    CURRENCY = "currency"


@dataclass(frozen=True, slots=True)
class Candles:
    """The window a chart draws, one entry per candle in each of the five
    arrays, handed to the function the `robot_traders_lab.indicators` entry
    point names.
    """

    open: npt.NDArray[np.float64]
    high: npt.NDArray[np.float64]
    low: npt.NDArray[np.float64]
    close: npt.NDArray[np.float64]
    volume: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ChartLine:
    """One series a strategy wants drawn beside its candles, returned by the
    function the `robot_traders_lab.indicators` entry point names.

    Attributes:
        name: The label the chart shows for the series.
        values: One entry per candle; a series that needs more history than
            the window holds leaves its first entries NaN, so every value
            stays under its own candle.
        colour: A CSS colour, `"blue"` or `"#1f77b4"`.
        pane: `"price"` draws the series over the candles, `"separate"` in a
            pane of its own.
        shape: `"line"` joins the values; `"histogram"` draws them as bars
            either side of zero, for a measure that reads as a distance from
            it, such as the gap between two averages.
    """

    name: str
    values: npt.NDArray[Any]
    colour: str
    pane: ChartPane = "price"
    shape: ChartShape = "line"
