from robottraderslab._core import (
    PerformanceReport,
    ReasonTable,
    ReportSection,
    ReportTable,
)

_TABLE_LABEL_WIDTH = 32
_TABLE_VALUE_WIDTH = 12


def format_performance_report(report: PerformanceReport) -> str:
    """The headline repeats figures the sections state, so a medium showing the
    whole report leaves it out.
    """
    blocks = [_format_section(section) for section in report.sections]
    blocks.append(format_table(report.trades))
    return "\n\n".join(blocks)


def format_reason_table(reason: ReasonTable) -> str:
    return _bar_separated(
        [reason.key, *reason.columns],
        [[row.label, *row.values] for row in reason.rows],
    )


def format_table(table: ReportTable) -> str:
    lines = [f"--- {table.title} ---", _table_line("", table.columns)]
    for group in table.groups:
        if group.title is not None:
            lines.extend(("", group.title))
        lines.extend(_table_line(row.label, row.values) for row in group.rows)
    return "\n".join(lines)


def _format_section(section: ReportSection) -> str:
    lines = [f"--- {section.title} ---"]
    lines.extend(f"{row.label}: {row.value}" for row in section.rows)
    return "\n".join(lines)


def _bar_separated(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for column, cell in enumerate(row):
            widths[column] = max(widths[column], len(cell))

    header_line = " | ".join(
        header.ljust(width) for header, width in zip(headers, widths)
    )
    separator = "-|-".join("-" * width for width in widths)
    table_rows = [
        " | ".join(cell.ljust(width) for cell, width in zip(row, widths))
        for row in rows
    ]
    return "\n".join([header_line, separator] + table_rows)


def _table_line(label: str, values: list[str]) -> str:
    cells = "".join(f"{value:>{_TABLE_VALUE_WIDTH}}" for value in values)
    return f"{label:<{_TABLE_LABEL_WIDTH}}{cells}".rstrip()
