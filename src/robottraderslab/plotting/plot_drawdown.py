from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from robottraderslab._core import DrawdownUnit

from .plot_base import FIGURE_SIZE, TITLE_FONT_SIZE, draw_drawdown, save_or_show


def plot_drawdown(
    drawdown: pd.Series,
    unit: DrawdownUnit,
    *,
    title: str | None,
    filename: str,
    save_to_path: Path | None,
) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_title(
        title or _default_title(float(drawdown.min()), unit), fontsize=TITLE_FONT_SIZE
    )
    draw_drawdown(ax, drawdown, unit)
    fig.tight_layout()
    save_or_show(fig, filename, save_to_path)


def _default_title(max_drawdown: float, unit: DrawdownUnit) -> str:
    match unit:
        case DrawdownUnit.PERCENT:
            figure = f"{max_drawdown:.2f}%"
        case DrawdownUnit.CURRENCY:
            figure = f"{max_drawdown:.2f}"
    return f"Drawdown Analysis | Max DD: {figure}"
