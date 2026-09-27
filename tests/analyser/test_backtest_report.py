import re
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from robottraderslab._core import OHLCVProviderProtocol
from robottraderslab.analyser import Analyser, AnalysisInputs
from robottraderslab.analyser.backtest_report import resolve_reports_dir
from robottraderslab.bootstrap import BotConfig, ReportConfig

_TRADES = pd.DataFrame(
    {
        "symbol": ["BTC/USDT:USDT", "BTC/USDT:USDT", "ETH/USDT:USDT", "ETH/USDT:USDT"],
        "side": ["long", "short", "long", "short"],
        "net_pnl": [100.0, -50.0, 150.0, -30.0],
        "net_pnl_pct": [0.01, -0.005, 0.015, -0.003],
        "entry_fee": [1.0, 2.0, 1.5, 1.8],
        "exit_fee": [1.0, 2.0, 1.5, 1.8],
        "fee": [2.0, 4.0, 3.0, 3.6],
        "entry_time": pd.to_datetime(
            [
                "2024-01-01 10:00:00",
                "2024-01-02 14:00:00",
                "2024-01-03 09:00:00",
                "2024-01-04 16:00:00",
            ]
        ),
        "exit_time": pd.to_datetime(
            [
                "2024-01-01 18:00:00",
                "2024-01-04 14:00:00",
                "2024-01-05 09:00:00",
                "2024-01-06 16:00:00",
            ]
        ),
        "entry_reason": ["signal", "signal", "manual", "signal"],
        "exit_reason": ["take_profit", "stop_loss", "take_profit", "signal"],
        "tag": ["setup_a", "setup_a", "setup_b", "setup_b"],
    }
)
_EQUITY_CURVE = pd.Series(
    [1000, 1100, 1050, 1200, 1150, 1300, 1250, 1150, 1100, 1450],
    index=pd.date_range("2024-01-01", periods=10, freq="D"),
    name="equity",
)
_BOT_TOML = '[strategy]\nstrategy_class = "test"\napi_key = "secret123"\n'


def _analyser(trades: pd.DataFrame) -> Analyser:
    inputs = AnalysisInputs(
        trades=trades,
        open_positions=pd.DataFrame(),
        equity_curve=_EQUITY_CURVE,
        initial_balance=float(_EQUITY_CURVE.iloc[0]),
    )
    return Analyser.from_analysis_inputs(
        inputs, Mock(spec=OHLCVProviderProtocol), ReportConfig()
    )


def _bot_config(tmp_path: Path, reports_dir: Path | None = None) -> BotConfig:
    toml_file = tmp_path / "bot-example.toml"
    overlay = None
    if reports_dir is not None:
        overlay = f'[report]\nreports_dir = "{reports_dir.as_posix()}"\n'
    toml_file.write_bytes(_BOT_TOML.encode("utf-8"))
    return BotConfig.from_file(toml_file, overlay=overlay)


class TestResolveReportsDir:
    def test_the_declared_directory_wins(self):
        bot_config = Mock(spec=BotConfig)
        bot_config.report = ReportConfig(reports_dir=Path("/declared"))
        bot_config.config_dir = Path("/beside/config")

        assert resolve_reports_dir(bot_config) == Path("/declared")

    def test_a_bot_naming_no_directory_keeps_it_beside_its_config(self):
        bot_config = Mock(spec=BotConfig)
        bot_config.report = ReportConfig()
        bot_config.config_dir = Path("/beside/config")

        assert resolve_reports_dir(bot_config) == Path("/beside/config/reports")

    def test_without_a_config_dir_the_reports_directory_is_relative(self):
        bot_config = Mock(spec=BotConfig)
        bot_config.report = ReportConfig()
        bot_config.config_dir = None

        assert resolve_reports_dir(bot_config) == Path("reports")


@pytest.fixture(scope="module")
def run_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One run folder, written once and only read by the tests sharing it."""
    reports_root = tmp_path_factory.mktemp("reports")
    bot_config = _bot_config(tmp_path_factory.mktemp("bot"), reports_root)
    return _analyser(_TRADES).save_run(bot_config, "test")


@pytest.fixture(scope="module")
def report_text(run_folder: Path) -> str:
    return (run_folder / "report.txt").read_text(encoding="utf-8")


class TestSaveRun:
    def test_the_folder_is_named_after_the_bot_and_the_moment_it_ran(self, run_folder):
        assert re.match(
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_backtest_bot-example", run_folder.name
        )

    def test_the_configuration_file_is_copied_byte_for_byte(self, run_folder):
        assert (run_folder / "bot-example.toml").read_bytes() == _BOT_TOML.encode(
            "utf-8"
        )

    def test_the_report_carries_no_configuration(self, report_text):
        assert "Configuration" not in report_text

    def test_the_performance_summary_is_included(self, report_text):
        assert "--- Performance Summary ---" in report_text
        assert "--- Overview ---" in report_text
        assert "PnL:" in report_text
        assert "Win rate" in report_text
        assert "Sharpe ratio:" in report_text

    def test_each_symbol_is_analysed(self, report_text):
        assert "--- Symbol Analysis: BTC/USDT:USDT ---" in report_text
        assert "--- Symbol Analysis: ETH/USDT:USDT ---" in report_text
        assert "SYMBOL ANALYSIS: BTC/USDT:USDT" in report_text
        assert "SYMBOL ANALYSIS: ETH/USDT:USDT" in report_text

    def test_a_missing_reports_directory_is_created(self, tmp_path):
        nested = tmp_path / "custom" / "nested"
        bot_config = _bot_config(tmp_path, nested)

        directory = _analyser(_TRADES).save_run(bot_config, "test")

        assert directory.parent == nested
        assert (directory / "report.txt").exists()

    def test_only_the_side_that_traded_gets_a_column(self, tmp_path):
        long_only = _TRADES[_TRADES["side"] == "long"]
        bot_config = _bot_config(tmp_path)

        directory = _analyser(long_only).save_run(bot_config, "test")

        report_text = (directory / "report.txt").read_text(encoding="utf-8")
        assert "All" + " " * 8 + "Long" in report_text
        assert "Short" not in report_text

    def test_the_reports_directory_sits_beside_the_config(self, tmp_path):
        bot_config = _bot_config(tmp_path)

        directory = _analyser(_TRADES).save_run(bot_config, "test")

        assert directory.parent == tmp_path / "reports"

    def test_a_config_naming_no_file_cannot_be_saved(self):
        bot_config = BotConfig.from_text(_BOT_TOML)

        with pytest.raises(ValueError, match="no configuration file"):
            _analyser(_TRADES).save_run(bot_config, "test")

    def test_a_config_declaring_no_profiles_writes_no_page(self, run_folder):
        assert not list(run_folder.glob("*.html"))

    def test_the_config_copy_carries_whatever_the_source_declares(self, tmp_path):
        """config.toml is copied byte for byte: a `[live]` section, and any
        `secret_name` reference in it, travels with the rest of the file.
        """
        toml_file = tmp_path / "bot-example.toml"
        toml_file.write_bytes(
            b'[strategy]\nstrategy_class = "test"\n\n'
            b"[live]\nohlcv_provider = {}\n"
            b'trading_account = { exchange = "bitget", secret_name = "demo" }\n'
        )
        bot_config = BotConfig.from_file(toml_file)

        directory = _analyser(_TRADES).save_run(bot_config, "test")

        copied = (directory / "bot-example.toml").read_text(encoding="utf-8")
        assert "[live]" in copied
        assert 'secret_name = "demo"' in copied
