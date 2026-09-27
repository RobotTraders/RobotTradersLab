from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from robottraderslab._core import (
    DrawdownUnit,
    PerformanceReport,
    ReportSection,
    ReportTable,
)

from .plot_base import (
    draw_drawdown,
    draw_equity_panel,
    save_or_show,
    validate_series_not_empty,
)

_TITLE = "Trading Performance Summary"
_FIGURE_SIZE = (15, 10)
_LEFT_MARGIN = 0.08
_RIGHT_MARGIN = 0.97
_TOP_MARGIN = 0.92
_BOTTOM_MARGIN = 0.04
_COLUMN_GAP = 0.24
_ROW_GAP = 0.20
_COLUMN_RATIOS = (2.0, 1.0)
_ROW_RATIOS = (1.0, 1.0)
_SUPTITLE_Y = 0.985
_SUPTITLE_FONT_SIZE = 18
_SIDEBAR_FONT_SIZE = 10
_SIDEBAR_TEXT_X = 0.08
_SIDEBAR_TEXT_Y = 0.99
_SIDEBAR_BOX = {"boxstyle": "round,pad=0.35", "facecolor": "lightgray", "alpha": 0.6}
_EQUITY_LINE_WIDTH = 2.5
_EQUITY_FILL_ALPHA = 0.25
_PRICE_LINE_WIDTH = 1.5
_DRAWDOWN_LINE_WIDTH = 2.5
_DRAWDOWN_FILL_ALPHA = 0.35


def plot_performance_summary(
    equity_curve: pd.Series,
    drawdown: pd.Series,
    drawdown_unit: DrawdownUnit,
    report: PerformanceReport,
    price_series: pd.Series | None,
    price_label: str | None,
    *,
    filename: str,
    save_to_path: Path | None,
) -> None:
    """The sidebar shows the figures as the report states them, so the dashboard
    agrees with every other medium showing the same run.
    """
    validate_series_not_empty(equity_curve, "equity_curve")
    fig = plt.figure(figsize=_FIGURE_SIZE)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=_COLUMN_RATIOS,
        height_ratios=_ROW_RATIOS,
        wspace=_COLUMN_GAP,
        hspace=_ROW_GAP,
    )
    draw_equity_panel(
        fig.add_subplot(grid[0, 0]),
        equity_curve,
        price_series,
        price_label,
        equity_line_width=_EQUITY_LINE_WIDTH,
        price_line_width=_PRICE_LINE_WIDTH,
        fill_alpha=_EQUITY_FILL_ALPHA,
    )
    draw_drawdown(
        fig.add_subplot(grid[1, 0]),
        drawdown,
        drawdown_unit,
        line_width=_DRAWDOWN_LINE_WIDTH,
        fill_alpha=_DRAWDOWN_FILL_ALPHA,
    )
    _draw_sidebar(fig.add_subplot(grid[:, 1]), report)
    fig.suptitle(
        _TITLE,
        fontsize=_SUPTITLE_FONT_SIZE,
        fontweight="bold",
        x=(_LEFT_MARGIN + _RIGHT_MARGIN) / 2.0,
        y=_SUPTITLE_Y,
    )
    fig.subplots_adjust(
        left=_LEFT_MARGIN, right=_RIGHT_MARGIN, top=_TOP_MARGIN, bottom=_BOTTOM_MARGIN
    )
    save_or_show(fig, filename, save_to_path)


def _draw_sidebar(ax: Axes, report: PerformanceReport) -> None:
    ax.axis("off")
    ax.text(
        _SIDEBAR_TEXT_X,
        _SIDEBAR_TEXT_Y,
        _sidebar_text(report),
        transform=ax.transAxes,
        fontsize=_SIDEBAR_FONT_SIZE,
        verticalalignment="top",
        bbox=_SIDEBAR_BOX,
    )


def _sidebar_text(report: PerformanceReport) -> str:
    blocks = [_section_block(section) for section in report.sections]
    blocks.append(_table_block(report.trades))
    return "\n\n".join(blocks)


def _section_block(section: ReportSection) -> str:
    lines = [section.title.upper()]
    lines.extend(f"{row.label}: {row.value}" for row in section.rows)
    return "\n".join(lines)


def _table_block(table: ReportTable) -> str:
    """The table's first column measures every trade at once. A sidebar is one
    column wide, so the columns measuring one side at a time are left to the
    media with room for a table.
    """
    lines = [table.title.upper()]
    for group in table.groups:
        if group.title is not None:
            lines.extend(("", group.title.upper()))
        lines.extend(f"{row.label}: {row.values[0]}" for row in group.rows)
    return "\n".join(lines)
