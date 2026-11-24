import subprocess
import sys
from pathlib import Path

import pytest

from robottraderslab import BotConfig, run_backtest

BOT_CONFIG = """\
[backtest]
initial_balance = { USDT = 1000.0 }
maker_fee_rate = 0.0
taker_fee_rate = 0.0
start_date = "2024-01-01"
end_date = "2024-04-01"
{logging}
[backtest.ohlcv_provider]
ohlcv_provider = "mock"

[strategy]
strategy_class = "{strategy_class}"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1d"
available_balance_ratio = 0.5
fast_ma_length = 3
slow_ma_length = 7
"""

BROKEN_STRATEGY = """\
from robottraderslab import Symbol


class BrokenStrategy:
    market_type = "futures"

    def __init__(self, **kwargs):
        pass

    async def setup(self, requirements):
        requirements.ohlcv.add(Symbol.create("BTC/USDT:USDT"), "1d")

    def generate_trading_signals(self, ohlcvs):
        raise ValueError("signal-broken-99")

    def book_trading_actions(self, *args):
        pass
"""


@pytest.fixture
def bot_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "first-bot"
    folder.mkdir()
    return folder


@pytest.fixture
def config_file(bot_folder: Path) -> Path:
    return _write_config(bot_folder, strategy_class="futures_ma")


def test_a_saved_run_keeps_its_log_in_its_run_folder(bot_folder, config_file):
    completed = _backtest(bot_folder)

    assert completed.returncode == 0, completed.stderr
    (run_folder,) = (bot_folder / "backtests").iterdir()
    assert "Backtest starting" in _read(run_folder / "backtest.log")
    assert not (bot_folder / "logs").exists()


def test_an_unsaved_run_shows_its_log_in_the_terminal(bot_folder, config_file):
    completed = _backtest(bot_folder, "--no-save")

    assert completed.returncode == 0, completed.stderr
    assert "Backtest starting" in completed.stdout


def test_a_run_that_raises_leaves_its_log_in_its_run_folder(bot_folder):
    (bot_folder / "broken_strategy.py").write_text(BROKEN_STRATEGY)
    _write_config(bot_folder, strategy_class="broken_strategy.BrokenStrategy")

    completed = _backtest(bot_folder, cwd=bot_folder)

    assert completed.returncode != 0
    (run_folder,) = (bot_folder / "backtests").iterdir()
    assert "signal-broken-99" in _read(run_folder / "backtest.log")


def test_a_notebook_run_saves_its_log_in_its_run_folder(bot_folder, config_file):
    outputs = run_backtest(BotConfig.from_file(config_file))

    outputs.create_analyser()

    (run_folder,) = (bot_folder / "backtests").iterdir()
    assert "Backtest starting" in _read(run_folder / "backtest.log")


def test_a_notebook_run_with_its_log_file_disabled(bot_folder):
    config_file = _write_config(
        bot_folder,
        strategy_class="futures_ma",
        logging="\n[backtest.logging]\nenable_file = false\n",
    )
    outputs = run_backtest(BotConfig.from_file(config_file))

    outputs.create_analyser()

    (run_folder,) = (bot_folder / "backtests").iterdir()
    assert not (run_folder / "backtest.log").exists()


def _write_config(folder: Path, strategy_class: str, logging: str = "") -> Path:
    config_file = folder / "bot.toml"
    config_file.write_text(
        BOT_CONFIG.replace("{strategy_class}", strategy_class).replace(
            "{logging}", logging
        )
    )
    return config_file


def _backtest(
    folder: Path, *options: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "robottraderslab",
            "backtest",
            str(folder),
            "--no-chart",
            *options,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_a_run_that_records_no_log_holds_none(config_file):
    outputs = run_backtest(BotConfig.from_file(config_file), record_log=False)

    assert outputs.run_log is None
