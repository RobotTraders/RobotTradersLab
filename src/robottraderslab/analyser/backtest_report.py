from pathlib import Path

import pandas as pd

from robottraderslab._core import BacktestReport, PerformanceReport
from robottraderslab.bootstrap import BotConfig

from .analysis_tables import build_reason_tables, build_symbol_analysis


def build_backtest_report(
    performance: PerformanceReport, trades: pd.DataFrame, initial_equity: float
) -> BacktestReport:
    return BacktestReport(
        performance=performance,
        reasons=build_reason_tables(trades),
        symbols=[
            build_symbol_analysis(trades, symbol, initial_equity)
            for symbol in sorted(trades["symbol"].unique())
        ],
    )


def resolve_reports_dir(bot_config: BotConfig) -> Path:
    """A bot declaring no directory of its own keeps its reports beside the
    config that produced them, so a run started from anywhere writes to the
    same place.
    """
    if bot_config.report.reports_dir is not None:
        return bot_config.report.reports_dir
    if bot_config.config_dir is not None:
        return bot_config.config_dir / "reports"
    return Path("reports")
