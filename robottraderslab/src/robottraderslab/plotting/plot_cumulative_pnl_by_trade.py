from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_base import (
    AXIS_LABEL_FONT_SIZE,
    BASELINE_ALPHA,
    FIGURE_SIZE,
    LINE_WIDTH,
    TITLE_FONT_SIZE,
    ColourPalette,
    apply_common_styling,
    save_or_show,
    validate_series_not_empty,
)

_DEFAULT_TITLE = "Cumulative PnL by Trade"
_BAR_WIDTH = 0.8


def plot_cumulative_pnl_by_trade(
    cumulative_pnl: pd.Series,
    *,
    show_percentage: bool,
    title: str | None,
    filename: str,
    save_to_path: Path | None,
) -> None:
    """Each bar is coloured by the trade it adds, so the first one takes the colour
    of its own result.
    """
    validate_series_not_empty(cumulative_pnl, "cumulative_pnl")
    per_trade = cumulative_pnl.diff().fillna(cumulative_pnl.iloc[0])
    colours = np.where(
        per_trade >= 0.0, ColourPalette.POSITIVE, ColourPalette.NEGATIVE
    ).tolist()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.bar(range(len(cumulative_pnl)), cumulative_pnl, color=colours, width=_BAR_WIDTH)
    ax.axhline(
        y=0, color=ColourPalette.NEUTRAL, linewidth=LINE_WIDTH, alpha=BASELINE_ALPHA
    )
    ax.set_title(title or _DEFAULT_TITLE, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Trade Number", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(
        _DEFAULT_TITLE + (" (%)" if show_percentage else ""),
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    apply_common_styling(ax)
    fig.tight_layout()
    save_or_show(fig, filename, save_to_path)
