from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter

from robottraderslab.plotting import (
    AXIS_LABEL_FONT_SIZE,
    TICK_FONT_SIZE,
    TITLE_FONT_SIZE,
    save_or_show,
)

from ..optimisation_result import SUMMARY_METRICS

_FIGURE_SIZE = (10, 8)
_SUMMARY_FIGURE_SIZE = (18, 10)
_SUMMARY_LAYOUT = (2, 3)
_SUMMARY_FILENAME = "heatmap_summary"
_SUMMARY_TITLE_FONT_SIZE = 16
_COLOUR_MAP = "RdYlGn"
_MAX_TICKS = 10
_LABEL_ROTATION = 45
_PROFILE_PREFIX = "strategy.profiles."


def plot_heatmap(
    df: pd.DataFrame,
    param1: str,
    param2: str,
    metric: str,
    *,
    smoothing: float = 0.0,
    title: str | None = None,
    filename: str | None = None,
    save_to_path: Path | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=_FIGURE_SIZE)
    _draw_heatmap(ax, df, param1, param2, metric, smoothing, title)
    fig.tight_layout()
    save_or_show(fig, filename or metric, save_to_path)
    return fig


def plot_heatmap_summary(
    df: pd.DataFrame,
    param1: str,
    param2: str,
    *,
    smoothing: float = 0.0,
    title: str | None = None,
    filename: str | None = None,
    save_to_path: Path | None = None,
) -> Figure:
    fig, axes = plt.subplots(*_SUMMARY_LAYOUT, figsize=_SUMMARY_FIGURE_SIZE)
    if title is not None:
        fig.suptitle(title, fontsize=_SUMMARY_TITLE_FONT_SIZE, fontweight="bold")
    for ax, metric in zip(axes.flat, SUMMARY_METRICS, strict=True):
        _draw_heatmap(ax, df, param1, param2, metric, smoothing, None)
    fig.tight_layout()
    save_or_show(fig, filename or _SUMMARY_FILENAME, save_to_path)
    return fig


def _draw_heatmap(
    ax: Axes,
    df: pd.DataFrame,
    param1: str,
    param2: str,
    metric: str,
    smoothing: float,
    title: str | None,
) -> None:
    """The first parameter runs down the rows from its largest value, so the grid
    reads like a chart with both axes growing away from the origin.
    """
    pivot = df.pivot_table(index=param1, columns=param2, values=metric).sort_index(
        ascending=False
    )
    cells = pivot.to_numpy(dtype=float)
    if smoothing > 0.0:
        cells = gaussian_filter(cells, sigma=smoothing)
    image = ax.imshow(cells, cmap=_COLOUR_MAP, aspect="auto")
    ax.figure.colorbar(image, ax=ax)
    x_step = max(1, len(pivot.columns) // _MAX_TICKS)
    y_step = max(1, len(pivot.index) // _MAX_TICKS)
    ax.set_xticks(
        range(0, len(pivot.columns), x_step),
        pivot.columns[::x_step],
        rotation=_LABEL_ROTATION,
        ha="right",
    )
    ax.set_yticks(range(0, len(pivot.index), y_step), pivot.index[::y_step])
    ax.set_xlabel(_short_name(param2), fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(_short_name(param1), fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title(
        title or _display_name(metric), fontsize=TITLE_FONT_SIZE, fontweight="bold"
    )
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)


def _display_name(metric: str) -> str:
    return metric.replace("_", " ").title()


def _short_name(param: str) -> str:
    return param.removeprefix(_PROFILE_PREFIX)
