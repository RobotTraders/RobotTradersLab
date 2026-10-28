from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from robottraderslab._core import OHLCVProviderProtocol
from robottraderslab.analyser import Analyser, AnalysisInputs
from robottraderslab.bootstrap import BotConfig, ReportConfig

RUN_MOMENT = "2099-01-02_03-04-05"
RUN_START = "2024-01-01"

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


def _bot_config(bot_folder: Path) -> BotConfig:
    toml_file = bot_folder / "bot.toml"
    toml_file.write_bytes(_BOT_TOML.encode("utf-8"))
    return BotConfig.from_file(toml_file)


@pytest.fixture(scope="module")
def run_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One run folder, written once and only read by the tests sharing it."""
    bot_config = _bot_config(tmp_path_factory.mktemp("bot"))
    return _analyser(_TRADES).save_run(bot_config, "test", RUN_MOMENT, RUN_START)


@pytest.fixture(scope="module")
def report_text(run_folder: Path) -> str:
    return (run_folder / "report.txt").read_text(encoding="utf-8")


class TestSaveRun:
    def test_the_folder_is_named_after_the_moment_it_ran(self, run_folder):
        assert run_folder.name == "2099-01-02_03-04-05"

    def test_a_folder_the_run_log_opened_is_completed_in_place(self, tmp_path):
        bot_config = _bot_config(tmp_path)
        opened = tmp_path / "backtests" / "2099-01-02_03-04-05"
        opened.mkdir(parents=True)
        (opened / "backtest.log").write_text("log-line-99", encoding="utf-8")

        directory = _analyser(_TRADES).save_run(
            bot_config, "test", RUN_MOMENT, RUN_START
        )

        assert directory == opened
        assert (directory / "backtest.log").read_text(encoding="utf-8") == "log-line-99"
        assert (directory / "report.txt").exists()

    def test_a_folder_another_run_completed_is_never_overwritten(self, tmp_path):
        bot_config = _bot_config(tmp_path)
        _analyser(_TRADES).save_run(bot_config, "test", RUN_MOMENT, RUN_START)

        with pytest.raises(FileExistsError, match="already holds a saved run"):
            _analyser(_TRADES).save_run(bot_config, "test", RUN_MOMENT, RUN_START)

    def test_the_configuration_file_is_copied_byte_for_byte(self, run_folder):
        assert (run_folder / "bot.toml").read_bytes() == _BOT_TOML.encode("utf-8")

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

    def test_only_the_side_that_traded_gets_a_column(self, tmp_path):
        long_only = _TRADES[_TRADES["side"] == "long"]
        bot_config = _bot_config(tmp_path)

        directory = _analyser(long_only).save_run(
            bot_config, "test", RUN_MOMENT, RUN_START
        )

        report_text = (directory / "report.txt").read_text(encoding="utf-8")
        assert "All" + " " * 8 + "Long" in report_text
        assert "Short" not in report_text

    def test_the_run_lands_under_the_bot_folder_backtests(self, tmp_path):
        bot_config = _bot_config(tmp_path)

        directory = _analyser(_TRADES).save_run(
            bot_config, "test", RUN_MOMENT, RUN_START
        )

        assert directory.parent == tmp_path / "backtests"

    def test_a_config_naming_no_file_cannot_be_saved(self):
        bot_config = BotConfig.from_text(f'name = "test-bot"\n{_BOT_TOML}')

        with pytest.raises(ValueError, match="no configuration file"):
            _analyser(_TRADES).save_run(bot_config, "test", RUN_MOMENT, RUN_START)

    def test_a_config_declaring_no_profiles_writes_no_page(self, run_folder):
        assert not list(run_folder.glob("*.html"))

    def test_the_config_copy_carries_whatever_the_source_declares(self, tmp_path):
        toml_file = tmp_path / "bot.toml"
        toml_file.write_bytes(
            b'[strategy]\nstrategy_class = "test"\n\n'
            b"[live]\nohlcv_provider = {}\n"
            b'trading_account = { exchange = "bitget", secret_name = "demo" }\n'
        )
        bot_config = BotConfig.from_file(toml_file)

        directory = _analyser(_TRADES).save_run(
            bot_config, "test", RUN_MOMENT, RUN_START
        )

        copied = (directory / "bot.toml").read_text(encoding="utf-8")
        assert "[live]" in copied
        assert 'secret_name = "demo"' in copied
