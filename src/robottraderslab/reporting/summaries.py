from collections.abc import Sequence
from io import TextIOWrapper

from robottraderslab._core import (
    PerformanceReport,
    ReasonTable,
    ReportTable,
    SymbolAnalysis,
)

from .text_format import format_performance_report, format_reason_table, format_table


def print_filtered_analysis(table: ReportTable) -> None:
    write_trades_summary(table, ())


def print_long_short_comparison(table: ReportTable) -> None:
    write_trades_summary(table, ())


def print_performance_summary(
    report: PerformanceReport, reasons: Sequence[ReasonTable]
) -> None:
    write_performance_summary(report, reasons)


def print_symbol_analysis(analysis: SymbolAnalysis) -> None:
    write_trades_summary(analysis.trades, analysis.reasons)


def write_performance_summary(
    report: PerformanceReport,
    reasons: Sequence[ReasonTable],
    output_file: TextIOWrapper | None = None,
) -> None:
    print(format_performance_report(report), file=output_file)
    _write_reason_tables(reasons, output_file)


def write_trades_summary(
    table: ReportTable,
    reasons: Sequence[ReasonTable],
    output_file: TextIOWrapper | None = None,
) -> None:
    print(format_table(table), file=output_file)
    _write_reason_tables(reasons, output_file)


def _write_reason_tables(
    reasons: Sequence[ReasonTable], output_file: TextIOWrapper | None
) -> None:
    for reason in reasons:
        if reason.rows:
            print(f"\n--- {reason.title} ---", file=output_file)
            print(format_reason_table(reason), file=output_file)
