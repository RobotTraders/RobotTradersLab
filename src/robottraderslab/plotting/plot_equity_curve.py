from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from robottraderslab._core import format_percent

from .plot_base import (
    FIGURE_SIZE,
    TITLE_FONT_SIZE,
    draw_equity_panel,
    save_or_show,
    validate_series_not_empty,
)

_UNMEASURED = "N/A"


def plot_equity_curve(
    equity_curve: pd.Series,
    roi: float | None,
    max_drawdown: float | None,
    price_series: pd.Series | None,
    price_label: str | None,
    *,
    title: str | None,
    filename: str,
    save_to_path: Path | None,
) -> None:
    validate_series_not_empty(equity_curve, "equity_curve")
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_title(title or _default_title(roi, max_drawdown), fontsize=TITLE_FONT_SIZE)
    draw_equity_panel(ax, equity_curve, price_series, price_label)
    fig.tight_layout()
    save_or_show(fig, filename, save_to_path)


def _default_title(roi: float | None, max_drawdown: float | None) -> str:
    roi_text = _UNMEASURED if roi is None else format_percent(roi)
    drawdown_text = (
        _UNMEASURED if max_drawdown is None else format_percent(max_drawdown)
    )
    return f"Portfolio Performance | ROI: {roi_text} | Max DD: {drawdown_text}"
