import shutil
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path

from robottraderslab._core import BacktestReport, SymbolAnalysis

from .summaries import write_performance_summary, write_trades_summary

_REPORT_FILENAME = "report.txt"
_BACKTESTS_DIR_NAME = "backtests"
_REPORTS_DIR_NAME = "reports"
_GRID_SEARCHES_DIR_NAME = "grid-searches"


def timestamp_now() -> str:
    """Return the current moment as `YYYY-MM-DD_HH-MM-SS`."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def backtest_folder(bot_folder: Path, timestamp: str) -> Path:
    return bot_folder / _BACKTESTS_DIR_NAME / timestamp


def grid_search_folder(bot_folder: Path, timestamp: str) -> Path:
    return bot_folder / _GRID_SEARCHES_DIR_NAME / timestamp


def report_folder(bot_folder: Path, timestamp: str) -> Path:
    return bot_folder / _REPORTS_DIR_NAME / timestamp


def create_run_folder(
    report: BacktestReport, directory: Path, config_file: Path
) -> None:
    """A folder the run's log already opened is completed in place, and a
    folder another run already completed is never overwritten.

    Raises:
        FileExistsError: If the folder already holds a run's report.
    """
    report_file = directory / _REPORT_FILENAME
    if report_file.exists():
        raise FileExistsError(f"`{directory}` already holds a saved run")
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_file, directory / config_file.name)
    _write_report(report, report_file)


def _write_report(report: BacktestReport, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write("--- Performance Summary ---\n")
        write_performance_summary(report.performance, report.reasons, output_file)
        for symbol in report.symbols:
            _write_symbol(output_file, symbol)


def _write_symbol(output_file: TextIOWrapper, analysis: SymbolAnalysis) -> None:
    output_file.write(f"\n--- Symbol Analysis: {analysis.symbol} ---\n")
    write_trades_summary(analysis.trades, analysis.reasons, output_file)
