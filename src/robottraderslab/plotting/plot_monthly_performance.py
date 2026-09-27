import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.container import BarContainer

from robottraderslab._core import FRACTION_TO_PERCENT, format_percent

from .plot_base import (
    AXIS_LABEL_FONT_SIZE,
    BASELINE_ALPHA,
    FIGURE_SIZE,
    TITLE_FONT_SIZE,
    ColourPalette,
    apply_common_styling,
    save_or_show,
)

_MONTHS = range(1, 13)
_MONTH_LABELS = [calendar.month_name[month] for month in _MONTHS]
_BAR_TEXT_FONT_SIZE = 9
_LABEL_ROTATION = 45
_HEADROOM = 0.15
_BELOW_BAR_OFFSET_POINTS = -2


def plot_monthly_performance(
    monthly_returns: pd.Series,
    year: int,
    *,
    title: str | None,
    filename: str,
    save_to_path: Path | None,
) -> None:
    """A month without a return, the year's first or one the run never reached,
    shows a flat bar so the year always reads as twelve.
    """
    returns = monthly_returns.fillna(0.0)
    percentages = (
        returns.set_axis(pd.DatetimeIndex(returns.index).month).reindex(
            _MONTHS, fill_value=0.0
        )
        * FRACTION_TO_PERCENT
    )
    cumulative = float(np.prod(1.0 + returns.to_numpy(dtype=float)) - 1.0)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bars = ax.bar(
        range(len(_MONTHS)),
        percentages,
        color=np.where(
            percentages >= 0.0, ColourPalette.POSITIVE, ColourPalette.NEGATIVE
        ).tolist(),
        linewidth=0,
    )
    _annotate_bars(ax, bars)
    ax.axhline(0, color=ColourPalette.NEUTRAL, alpha=BASELINE_ALPHA)
    ax.set_title(
        title or f"{year} (Cumulative Performance: {format_percent(cumulative)})",
        fontsize=TITLE_FONT_SIZE,
    )
    ax.set_ylabel("Performance (%)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticks(range(len(_MONTHS)), _MONTH_LABELS, rotation=_LABEL_ROTATION)
    low, high = float(percentages.min()), float(percentages.max())
    headroom = (high - low) * _HEADROOM
    ax.set_ylim(low - headroom, high + headroom)
    apply_common_styling(ax)
    fig.tight_layout()
    save_or_show(fig, filename, save_to_path)


def _annotate_bars(ax: Axes, bars: BarContainer) -> None:
    for bar in bars:
        height = float(bar.get_height())
        above = height >= 0.0
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 0 if above else _BELOW_BAR_OFFSET_POINTS),
            textcoords="offset points",
            ha="center",
            va="bottom" if above else "top",
            fontsize=_BAR_TEXT_FONT_SIZE,
            color=ColourPalette.NEUTRAL,
        )
