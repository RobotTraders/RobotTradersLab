from dataclasses import dataclass
from typing import Literal

type CalculationMethod = Literal["daily", "tradingview_like"]

DEFAULT_ANNUALIZATION_FACTOR = 365.0
DEFAULT_CALCULATION_METHOD: CalculationMethod = "daily"
DEFAULT_RISK_FREE_RATE = 0.0


@dataclass(frozen=True, slots=True)
class ReportRow:
    """One figure of the performance report, ready to be shown as it stands."""

    label: str
    value: str


@dataclass(frozen=True, slots=True)
class ReportSection:
    """A titled group of report figures."""

    title: str
    rows: list[ReportRow]


@dataclass(frozen=True, slots=True)
class ReportTableRow:
    """One figure measured per column, aligned with its table's columns."""

    label: str
    values: list[str]


@dataclass(frozen=True, slots=True)
class ReportGroup:
    """A titled run of table rows.

    The rows a table opens on need no name of their own, so the first group
    may go untitled.
    """

    title: str | None
    rows: list[ReportTableRow]


@dataclass(frozen=True, slots=True)
class ReportTable:
    """A titled table of figures, each measured over every column at once."""

    title: str
    columns: list[str]
    groups: list[ReportGroup]


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """How a run performed, described once for every medium that shows it.

    The figures arrive already formatted, so printing it, writing it to a file
    and rendering it as HTML all show the same thing. The headline answers
    "was this any good" on its own, for a medium with room to show a few
    figures before the rest; the sections and the trades table hold the whole
    account.
    """

    headline: list[ReportRow]
    sections: list[ReportSection]
    trades: ReportTable


@dataclass(frozen=True, slots=True)
class ReasonTable:
    """Every figure arrives formatted, and `key` heads the column the row
    labels fall under.
    """

    title: str
    key: str
    columns: list[str]
    rows: list[ReportTableRow]


@dataclass(frozen=True, slots=True)
class SymbolAnalysis:
    symbol: str
    trades: ReportTable
    reasons: list[ReasonTable]


@dataclass(frozen=True, slots=True)
class BacktestReport:
    performance: PerformanceReport
    reasons: list[ReasonTable]
    symbols: list[SymbolAnalysis]
