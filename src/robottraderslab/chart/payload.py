from dataclasses import dataclass
from enum import StrEnum

from robottraderslab._core import DrawdownUnit, PerformanceReport

type Rows = list[list[float]]


class Pane(StrEnum):
    """Which pane of the price chart an indicator belongs on."""

    PRICE = "price"
    SEPARATE = "separate"


class Shape(StrEnum):
    """How an indicator is drawn: as a curve, or as bars either side of zero."""

    LINE = "line"
    HISTOGRAM = "histogram"


class MarkerSide(StrEnum):
    """Which side of the book an executed order fell on."""

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class ChartSeries:
    """A named indicator drawn on the price pane or on a pane of its own.

    Each point is a row of the time in whole epoch seconds and the value.
    """

    name: str
    pane: Pane
    shape: Shape
    colour: str
    points: Rows


@dataclass(frozen=True, slots=True)
class TradeMarker:
    """An executed entry or exit drawn against the candles."""

    time: int
    side: MarkerSide
    label: str


@dataclass(frozen=True, slots=True)
class ProfileSetting:
    """One configured value of the profile a link leads to."""

    name: str
    value: str


@dataclass(frozen=True, slots=True)
class ChartLink:
    """A sibling chart reachable from this one, by a path relative to it."""

    label: str
    href: str
    current: bool
    settings: list[ProfileSetting]


@dataclass(frozen=True, slots=True)
class TradeRow:
    """One trade as listed in the table under the charts.

    A trade still open carries no exit, so the fields an exit fills in are
    absent.
    """

    entry_time: int
    exit_time: int | None
    side: str
    entry_price: float
    exit_price: float | None
    net_pnl: float
    net_pnl_pct: float
    entry_reason: str
    exit_reason: str | None


@dataclass(frozen=True, slots=True)
class MarketView:
    """One market's candles, indicators, trade markers and trade rows.

    Candles and curves are rows led by the time in whole epoch seconds, a
    candle's followed by its open, high, low and close and a curve's by its
    value, so a long history travels without a name beside every figure.
    """

    label: str
    settings: list[ProfileSetting]
    candles: Rows
    series: list[ChartSeries]
    markers: list[TradeMarker]
    trades: list[TradeRow]


@dataclass(frozen=True, slots=True)
class ChartPayload:
    """The same payload shape is produced by a backtest and by a live account, so
    the renderer never learns which one it is drawing. A page with several
    markets switches between them in place; a page with siblings of its own
    links to them instead, and carries one market of its own.
    """

    title: str
    markets: list[MarketView]
    report: PerformanceReport
    equity: Rows
    drawdown: Rows
    drawdown_unit: DrawdownUnit
    links: list[ChartLink]
    configuration: str
