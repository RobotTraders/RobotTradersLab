import shutil
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path

from robottraderslab._core import BacktestReport, SymbolAnalysis

from .summaries import write_performance_summary, write_trades_summary

_REPORT_FILENAME = "report.txt"


def timestamp_now() -> str:
    """Return the current moment as `YYYY-MM-DD_HH-MM-SS`."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def run_folder_name(bot_stem: str, timestamp: str) -> str:
    """Name a run folder as `<timestamp>_backtest_<bot_stem>`."""
    return f"{timestamp}_backtest_{bot_stem}"


def create_run_folder(
    report: BacktestReport, reports_root: Path, folder_name: str, config_file: Path
) -> Path:
    """
    Args:
        report: The measured performance and per-symbol breakdown to write.
        config_file: The configuration file, copied into the folder byte for
            byte, under its own name.

    Returns:
        The created folder.
    """
    directory = reports_root / folder_name
    directory.mkdir(parents=True)
    shutil.copyfile(config_file, directory / config_file.name)
    _write_report(report, directory / _REPORT_FILENAME)
    return directory


def _write_report(report: BacktestReport, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write("--- Performance Summary ---\n")
        write_performance_summary(report.performance, report.reasons, output_file)
        for symbol in report.symbols:
            _write_symbol(output_file, symbol)


def _write_symbol(output_file: TextIOWrapper, analysis: SymbolAnalysis) -> None:
    output_file.write(f"\n--- Symbol Analysis: {analysis.symbol} ---\n")
    write_trades_summary(analysis.trades, analysis.reasons, output_file)
