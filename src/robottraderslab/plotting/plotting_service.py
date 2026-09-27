from pathlib import Path

import pandas as pd

from robottraderslab._core import DrawdownUnit, PerformanceReport, last_per_month

from .plot_cumulative_pnl_by_trade import plot_cumulative_pnl_by_trade
from .plot_drawdown import plot_drawdown
from .plot_equity_curve import plot_equity_curve
from .plot_monthly_performance import plot_monthly_performance
from .plot_performance_summary import plot_performance_summary


class PlottingService:
    """Draws a run's matplotlib figures, saving each one where a path is given."""

    def __init__(self, save_to_path: Path | None) -> None:
        """
        Args:
            save_to_path: Directory each figure is written to; with none given,
                every figure is shown on screen.
        """
        self._save_to_path = save_to_path

    def plot_cumulative_pnl_by_trade(
        self,
        cumulative_pnl: pd.Series,
        *,
        show_percentage: bool,
        title: str | None = None,
        filename: str | None = None,
    ) -> None:
        plot_cumulative_pnl_by_trade(
            cumulative_pnl,
            show_percentage=show_percentage,
            title=title,
            filename=filename or "cumulative_pnl_by_trade",
            save_to_path=self._save_to_path,
        )

    def plot_drawdown(
        self,
        drawdown: pd.Series,
        unit: DrawdownUnit,
        title: str | None = None,
        filename: str | None = None,
    ) -> None:
        plot_drawdown(
            drawdown,
            unit,
            title=title,
            filename=filename or "drawdown",
            save_to_path=self._save_to_path,
        )

    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        roi: float | None,
        max_drawdown: float | None,
        price_series: pd.Series | None = None,
        title: str | None = None,
        filename: str | None = None,
        price_label: str | None = None,
    ) -> None:
        plot_equity_curve(
            equity_curve,
            roi,
            max_drawdown,
            price_series,
            price_label,
            title=title,
            filename=filename or "equity_curve",
            save_to_path=self._save_to_path,
        )

    def plot_monthly_performance(
        self,
        equity_curve: pd.Series,
        year: int,
        title: str | None = None,
        filename: str | None = None,
    ) -> None:
        plot_monthly_performance(
            _monthly_returns(equity_curve, year),
            year,
            title=title,
            filename=filename or f"monthly_performance_{year}",
            save_to_path=self._save_to_path,
        )

    def plot_performance_summary(
        self,
        equity_curve: pd.Series,
        *,
        drawdown: pd.Series,
        drawdown_unit: DrawdownUnit,
        report: PerformanceReport,
        filename: str | None = None,
        price_series: pd.Series | None = None,
        price_label: str | None = None,
    ) -> None:
        plot_performance_summary(
            equity_curve,
            drawdown,
            drawdown_unit,
            report,
            price_series,
            price_label,
            filename=filename or "performance_summary",
            save_to_path=self._save_to_path,
        )


def _monthly_returns(equity_curve: pd.Series, year: int) -> pd.Series:
    """The first month of the year has no month before it to be measured from, so
    its return is left unstated.

    Raises:
        ValueError: If the curve has no point in the year.
    """
    monthly = last_per_month(equity_curve)
    that_year = monthly[pd.DatetimeIndex(monthly.index).year == year]
    if that_year.empty:
        raise ValueError(f"The equity curve has no point in {year}")
    return that_year.pct_change()
