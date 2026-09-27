from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from robottraderslab._core import DrawdownUnit

FIGURE_SIZE = (9, 4)
DPI = 300
TITLE_FONT_SIZE = 13
AXIS_LABEL_FONT_SIZE = 11
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
GRID_ALPHA = 0.3
LINE_WIDTH = 1.0
FILL_ALPHA = 0.2
BASELINE_ALPHA = 0.3
_AXIS_PADDING = 0.1


class ColourPalette:
    """Centralised colour definitions for consistent styling."""

    EQUITY = "#089981"
    DRAWDOWN = "indianred"
    PRICE = "#800080"
    POSITIVE = "#089981"
    NEGATIVE = "#F23645"
    NEUTRAL = "black"


def apply_common_styling(ax: Axes) -> None:
    ax.grid(True, alpha=GRID_ALPHA)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)


def draw_drawdown(
    ax: Axes,
    drawdown: pd.Series,
    unit: DrawdownUnit,
    *,
    line_width: float = LINE_WIDTH,
    fill_alpha: float = FILL_ALPHA,
) -> None:
    """The drawdown arrives already counted in the unit stated beside it."""
    ax.plot(drawdown, color=ColourPalette.DRAWDOWN, linewidth=line_width)
    ax.fill_between(
        drawdown.index, drawdown, alpha=fill_alpha, color=ColourPalette.DRAWDOWN
    )
    ax.axhline(
        y=0, color=ColourPalette.NEUTRAL, alpha=BASELINE_ALPHA, linewidth=LINE_WIDTH
    )
    ax.set_ylabel(drawdown_axis_label(unit), fontsize=AXIS_LABEL_FONT_SIZE)
    apply_common_styling(ax)
    fit_dates(ax, drawdown)


def draw_equity_panel(
    ax_price: Axes,
    equity_curve: pd.Series,
    price_series: pd.Series | None,
    price_label: str | None,
    *,
    equity_line_width: float = LINE_WIDTH,
    price_line_width: float = LINE_WIDTH,
    fill_alpha: float = FILL_ALPHA,
) -> None:
    """A price and an equity are counted in different ranges, so the price keeps
    the panel's own axis and the equity takes a twin axis on the right.
    """
    if price_series is None:
        _draw_equity(ax_price, equity_curve, equity_line_width, fill_alpha)
    else:
        ax_equity = ax_price.twinx()
        _draw_equity(ax_equity, equity_curve, equity_line_width, fill_alpha)
        _draw_price(ax_price, price_series, price_label, price_line_width)
        ax_equity.yaxis.label.set_color(ColourPalette.EQUITY)
        ax_equity.tick_params(axis="y", colors=ColourPalette.EQUITY)
        handles, labels = ax_price.get_legend_handles_labels()
        ax_equity.legend(handles, labels, loc="upper left", fontsize=LEGEND_FONT_SIZE)
    apply_common_styling(ax_price)
    fit_dates(ax_price, equity_curve)


def drawdown_axis_label(unit: DrawdownUnit) -> str:
    match unit:
        case DrawdownUnit.PERCENT:
            return "Drawdown %"
        case DrawdownUnit.CURRENCY:
            return "Drawdown in Quote"


def fit_dates(ax: Axes, curve: pd.Series) -> None:
    ax.set_xlim(curve.index.min(), curve.index.max())
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def save_or_show(fig: Figure, filename: str, save_to_path: Path | None) -> None:
    """A run with nowhere to write its figures is being watched, so each is shown."""
    if save_to_path is None:
        plt.show()
    else:
        save_to_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to_path / f"{filename}.png", dpi=DPI)
    plt.close(fig)


def validate_series_not_empty(series: pd.Series, name: str) -> None:
    if series.empty:
        raise ValueError(f"{name} series cannot be empty")


def _draw_equity(
    ax: Axes, equity_curve: pd.Series, line_width: float, fill_alpha: float
) -> None:
    ax.plot(equity_curve, color=ColourPalette.EQUITY, linewidth=line_width)
    ax.fill_between(
        equity_curve.index, equity_curve, alpha=fill_alpha, color=ColourPalette.EQUITY
    )
    ax.axhline(
        y=float(equity_curve.iloc[0]),
        color=ColourPalette.NEUTRAL,
        linewidth=LINE_WIDTH,
        linestyle="--",
    )
    ax.set_ylabel("Equity Value in Quote", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.set_ylim(*_padded(equity_curve.min(), equity_curve.max()))


def _draw_price(
    ax: Axes, price_series: pd.Series, price_label: str | None, line_width: float
) -> None:
    ax.plot(
        price_series, color=ColourPalette.PRICE, linewidth=line_width, label=price_label
    )
    ax.set_ylabel(
        "Asset Price in Quote", color=ColourPalette.PRICE, fontsize=AXIS_LABEL_FONT_SIZE
    )
    ax.tick_params(axis="y", colors=ColourPalette.PRICE, labelsize=TICK_FONT_SIZE)
    ax.set_ylim(*_padded(price_series.min(), price_series.max()))


def _padded(low: float, high: float) -> tuple[float, float]:
    span = high - low
    return low - span * _AXIS_PADDING, high + span * _AXIS_PADDING
