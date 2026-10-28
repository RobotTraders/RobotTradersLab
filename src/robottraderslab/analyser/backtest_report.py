from pathlib import Path

import pandas as pd

from robottraderslab._core import BacktestReport, PerformanceReport
from robottraderslab.bootstrap import BotConfig
from robottraderslab.reporting import backtest_folder

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


def resolve_run_folder(bot_config: BotConfig, timestamp: str) -> Path:
    """A run that logs into this folder opens it before the run starts, and the
    reports complete it afterwards, so both name it from the same `timestamp`,
    in the format `timestamp_now` gives.
    """
    return backtest_folder(bot_config.config_dir or Path(), timestamp)
