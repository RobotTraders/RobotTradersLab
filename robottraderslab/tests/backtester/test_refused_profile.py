import logging
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from robottraderslab.cli import main

BOT_CONFIG = """\
[backtest]
initial_balance = { USDT = 1000.0 }
maker_fee_rate = 0.0
taker_fee_rate = 0.0
start_date = "2024-01-01"
end_date = "2024-04-01"

[backtest.ohlcv_provider]
ohlcv_provider = "mock"

[strategy]
strategy_class = "crossing_strategy.CrossingStrategy"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1d"
fast = 3
slow = 7

[[strategy.profiles]]
symbol = "ETH/USDT:USDT"
timeframe = "1d"
fast = 50
slow = 30
"""

UNKNOWN_KEY_CONFIG = BOT_CONFIG.replace("fast = 50", "lenght = 5\nfast = 5")
KEY_ERROR_CONFIG = BOT_CONFIG.replace("fast = 50", "fast = 5").replace(
    "slow = 30", "slow = 30\nlookup = true"
)

CROSSING_STRATEGY = """\
from dataclasses import dataclass
from typing import ClassVar

from robottraderslab.exceptions import ProfileConfigError
from robottraderslab.strategies import Profile, ProfileStrategy


@dataclass(frozen=True, kw_only=True)
class CrossingProfile(Profile):
    fast: int
    slow: int
    lookup: bool = False

    def __post_init__(self) -> None:
        if self.lookup:
            raise KeyError("lookup")
        if self.fast >= self.slow:
            raise ProfileConfigError(
                f"fast ({self.fast}) must be below slow ({self.slow})"
            )


class CrossingStrategy(ProfileStrategy[CrossingProfile]):
    market_type: ClassVar[str] = "futures"
"""


@pytest.fixture
def create_bot_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[str], Path]]:
    def _create(config: str) -> Path:
        folder = tmp_path / "first-bot"
        folder.mkdir()
        (folder / "crossing_strategy.py").write_text(CROSSING_STRATEGY)
        (folder / "bot.toml").write_text(config)
        monkeypatch.syspath_prepend(str(folder))
        return folder

    yield _create
    sys.modules.pop("crossing_strategy", None)


@pytest.fixture
def bot_folder(create_bot_folder: Callable[[str], Path]) -> Path:
    return create_bot_folder(BOT_CONFIG)


@pytest.fixture
def keep_pytest_log_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    """`main` ends by shutting logging down, which closes pytest's own handlers."""
    monkeypatch.setattr(logging, "shutdown", lambda *_: None)


def _run_backtest(bot_folder: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(
        sys, "argv", ["rtlab", "backtest", str(bot_folder), "--no-chart", "--no-save"]
    )
    with pytest.raises(SystemExit) as exit_info:
        main()
    return exit_info.value.code


@pytest.mark.usefixtures("restore_logging", "keep_pytest_log_handlers")
def test_a_backtest_exits_78_naming_the_file_and_the_profile_its_class_refuses(
    bot_folder, monkeypatch, capsys
):
    exit_code = _run_backtest(bot_folder, monkeypatch)

    assert exit_code == 78
    assert capsys.readouterr().err.strip() == (
        f"Error: `{bot_folder / 'bot.toml'}`: CrossingStrategy: "
        "profile 2 (ETH/USDT:USDT 1d): fast (50) must be below slow (30)"
    )


@pytest.mark.usefixtures("restore_logging", "keep_pytest_log_handlers")
@pytest.mark.parametrize(
    ("config", "exit_code", "sentence"),
    [
        (
            UNKNOWN_KEY_CONFIG,
            78,
            "profile 2 (ETH/USDT:USDT 1d): lenght: unknown key",
        ),
        (KEY_ERROR_CONFIG, 1, "'lookup'"),
    ],
    ids=["unknown key", "KeyError raised by __post_init__"],
)
def test_a_backtest_keeps_the_ending_of_a_section_the_engine_cannot_build(
    create_bot_folder, monkeypatch, capsys, config, exit_code, sentence
):
    bot_folder = create_bot_folder(config)

    backtest_exit_code = _run_backtest(bot_folder, monkeypatch)

    assert backtest_exit_code == exit_code
    assert sentence in capsys.readouterr().err
