import asyncio
import subprocess
import sys
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from robottraderslab.bootstrap import BotConfig, bot_file
from robottraderslab.exceptions import BotCriticalError
from robottraderslab.live.report import main as report
from robottraderslab.scheduler import run_scheduler

BACKTEST_BOT = """\
[backtest]
initial_balance = { USDT = 1000.0 }
maker_fee_rate = 0.0
taker_fee_rate = 0.0
start_date = "2024-01-01"
end_date = "2024-04-01"

[backtest.ohlcv_provider]
ohlcv_provider = "mock"

[strategy]
strategy_class = "futures_ma"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1d"
available_balance_ratio = 0.5
fast_ma_length = 3
slow_ma_length = 7
"""

LIVE_BOT = """\
[live.ohlcv_provider]
ohlcv_provider = "mock"

[live.trading_account]
exchange = "simulator"
account_name = "account"
initial_balance = { USDT = 10000.0 }
maker_fee_rate = 0.001
taker_fee_rate = 0.002

[strategy]
strategy_class = "futures_ma"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1h"
available_balance_ratio = 0.25
fast_ma_length = 3
slow_ma_length = 7
"""

SECRETS = """\
[[secrets]]
name = "secret-entry-99"
exchange = "simulator"
"""

ONE_HOUR_CLOSE = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / ".rtlab").mkdir(parents=True)
    return root


@pytest.fixture
def backtest_bot(workspace: Path) -> Path:
    return _bot(workspace / "first-bot", BACKTEST_BOT)


@pytest.fixture
def live_bot(workspace: Path) -> Path:
    return _bot(workspace / "first-bot", LIVE_BOT)


@pytest.fixture
def browser() -> Iterator[Mock]:
    with patch("webbrowser.open", autospec=True) as opened:
        yield opened


@pytest.fixture
def default_event_loop_policy() -> Iterator[None]:
    """The scheduler launches its bots as subprocesses, which the default event
    loop supports on every platform, whatever policy an earlier test set.
    """
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(None)
    yield
    asyncio.set_event_loop_policy(previous)


@pytest.fixture
def one_hour_close() -> Iterator[Mock]:
    with patch("robottraderslab.scheduler.scheduler.datetime") as clock:
        clock.now.return_value = ONE_HOUR_CLOSE
        yield clock


class TestBacktestFromTheFolder:
    def test_a_saved_run_lands_under_backtests(self, backtest_bot):
        completed = _rtlab("backtest", backtest_bot, "--no-chart")

        assert completed.returncode == 0, completed.stderr
        (run_folder,) = (backtest_bot / "backtests").iterdir()
        assert sorted(path.name for path in run_folder.iterdir()) == [
            "backtest.log",
            "bot.toml",
            "report.html",
            "report.txt",
        ]

    def test_an_unsaved_run_writes_nothing(self, backtest_bot):
        completed = _rtlab("backtest", backtest_bot, "--no-chart", "--no-save")

        assert completed.returncode == 0, completed.stderr
        assert [path.name for path in backtest_bot.iterdir()] == ["bot.toml"]


class TestLiveReportIsKept:
    def test_a_saved_report_lands_under_reports_and_opens(self, live_bot, browser):
        report(live_bot, days=30, save=True)

        (report_folder,) = (live_bot / "reports").iterdir()
        page = report_folder / "report.html"
        assert page.is_file()
        browser.assert_called_once_with(page.as_uri())

    def test_an_unsaved_report_only_opens(self, live_bot, browser):
        report(live_bot, days=30, save=False)

        assert not (live_bot / "reports").exists()
        (opened,) = browser.call_args.args
        assert opened.endswith("/report.html")


class TestBotInASubfolder:
    def test_the_workspace_secrets_file_is_found(self, workspace):
        folder = _bot(workspace / "kraken" / "trend", LIVE_BOT)
        (workspace / "secrets.toml").write_text(SECRETS, encoding="utf-8")

        bot_config = BotConfig.from_file(bot_file(folder))

        assert bot_config.secrets_file == str(workspace / "secrets.toml")

    def test_the_nearest_secrets_file_wins(self, workspace):
        folder = _bot(workspace / "kraken" / "trend", LIVE_BOT)
        (workspace / "secrets.toml").write_text(SECRETS, encoding="utf-8")
        (workspace / "kraken" / "secrets.toml").write_text(SECRETS, encoding="utf-8")

        bot_config = BotConfig.from_file(bot_file(folder))

        assert bot_config.secrets_file == str(workspace / "kraken" / "secrets.toml")

    def test_no_secrets_file_is_sought_above_the_workspace(self, workspace):
        folder = _bot(workspace / "kraken" / "trend", LIVE_BOT)
        (workspace.parent / "secrets.toml").write_text(SECRETS, encoding="utf-8")

        bot_config = BotConfig.from_file(bot_file(folder))

        assert bot_config.secrets_file is None


class TestRegistryNamesTheFolder:
    def test_the_bot_runs_on_its_close_and_writes_inside_its_folder(
        self,
        workspace,
        live_bot,
        one_hour_close,
        default_event_loop_policy,
        names_in_tool_directory,
    ):
        registry = workspace / "registry.toml"
        registry.write_text(
            '[[bot]]\nconfig = "first-bot"\ntimeframes = ["1h"]\n', encoding="utf-8"
        )

        failures = run_scheduler(registry)

        assert failures == 0
        assert (live_bot / "logs" / "live.log").is_file()
        assert (live_bot / ".rtlab" / "bot.lock").is_file()
        assert names_in_tool_directory(workspace) == [
            "registry.launches.json",
            "registry.lock",
        ]


class TestRegistryInASubfolderOfTheWorkspace:
    @pytest.fixture
    def scheduled_bot_folder(self, workspace: Path, one_hour_close: Mock) -> Path:
        (workspace / "secrets.toml").write_text(SECRETS, encoding="utf-8")
        folder = _venue_bot(workspace / "kraken" / "trend")
        registry = workspace / "kraken" / "registry.toml"
        registry.write_text(
            '[[bot]]\nconfig = "trend"\ntimeframes = ["1d"]\n', encoding="utf-8"
        )
        run_scheduler(registry)
        return folder

    def test_the_scheduler_keeps_its_lock_in_the_tool_directory_of_the_workspace(
        self, workspace, scheduled_bot_folder
    ):
        assert (workspace / ".rtlab" / "kraken" / "registry.lock").is_file()

    def test_the_scheduler_makes_no_tool_directory_beside_its_registry(
        self, workspace, scheduled_bot_folder
    ):
        assert not (workspace / "kraken" / ".rtlab").exists()

    def test_a_bot_of_the_registry_keeps_the_store_of_the_workspace(
        self, workspace, scheduled_bot_folder
    ):
        loaded = BotConfig.from_file(bot_file(scheduled_bot_folder))

        assert loaded.live.ohlcv_provider["storage_dir"] == str(
            workspace / "data" / "ohlcvs"
        )

    def test_a_bot_of_the_registry_keeps_the_secrets_file_of_the_workspace(
        self, workspace, scheduled_bot_folder
    ):
        loaded = BotConfig.from_file(bot_file(scheduled_bot_folder))

        assert loaded.secrets_file == str(workspace / "secrets.toml")


class TestWorkspaceOfBotsThatHaveRun:
    @pytest.fixture(params=[False, True], ids=["first load", "after a cycle"])
    def cycle_has_run(self, request: pytest.FixtureRequest) -> bool:
        return request.param

    @pytest.fixture
    def workspace_bot_folders(
        self,
        workspace: Path,
        cycle_has_run: bool,
        create_lock_file: Callable[[Path], None],
    ) -> list[Path]:
        (workspace / "secrets.toml").write_text(SECRETS, encoding="utf-8")
        folders = [_venue_bot(workspace / name) for name in ("a", "b")]
        if cycle_has_run:
            for folder in folders:
                create_lock_file(bot_file(folder))
        return folders

    @pytest.fixture
    def solo_bot_folder(
        self,
        tmp_path: Path,
        cycle_has_run: bool,
        create_lock_file: Callable[[Path], None],
    ) -> Path:
        (tmp_path / "secrets.toml").write_text(SECRETS, encoding="utf-8")
        folder = _venue_bot(tmp_path / "solo")
        if cycle_has_run:
            create_lock_file(bot_file(folder))
        return folder

    def test_bots_of_a_workspace_share_its_store(
        self, workspace, workspace_bot_folders
    ):
        loaded = [BotConfig.from_file(bot_file(f)) for f in workspace_bot_folders]

        stores = {bot.live.ohlcv_provider["storage_dir"] for bot in loaded}

        assert stores == {str(workspace / "data" / "ohlcvs")}

    def test_bots_of_a_workspace_share_its_secrets_file(
        self, workspace, workspace_bot_folders
    ):
        loaded = [BotConfig.from_file(bot_file(f)) for f in workspace_bot_folders]

        secrets_files = {bot.secrets_file for bot in loaded}

        assert secrets_files == {str(workspace / "secrets.toml")}

    def test_a_bot_in_a_folder_of_its_own_keeps_its_store(self, solo_bot_folder):
        loaded = BotConfig.from_file(bot_file(solo_bot_folder))

        assert loaded.live.ohlcv_provider["storage_dir"] == str(
            solo_bot_folder / "data" / "ohlcvs"
        )

    def test_a_bot_in_a_folder_of_its_own_keeps_its_secrets_file(
        self, tmp_path, solo_bot_folder
    ):
        loaded = BotConfig.from_file(bot_file(solo_bot_folder))

        assert loaded.secrets_file == str(tmp_path / "secrets.toml")


class TestFolderNamesTheBot:
    def test_a_bot_setting_no_name_takes_its_folder_name(self, live_bot):
        bot_config = BotConfig.from_file(bot_file(live_bot))

        assert bot_config.name == "first-bot"

    def test_renaming_the_folder_renames_the_bot(self, live_bot):
        renamed = live_bot.rename(live_bot.with_name("eth-trend"))

        bot_config = BotConfig.from_file(bot_file(renamed))

        assert bot_config.name == "eth-trend"

    def test_a_declared_name_is_refused(self, workspace):
        folder = _bot(workspace / "first-bot", f'name = "alpha"\n{LIVE_BOT}')

        with pytest.raises(BotCriticalError, match="named after its folder"):
            BotConfig.from_file(bot_file(folder))


class TestPathToATomlFile:
    def test_the_command_names_the_folder_to_pass(self, workspace):
        config = workspace / "trend" / "bot_b.toml"
        config.parent.mkdir()
        config.write_text(BACKTEST_BOT, encoding="utf-8")

        completed = _rtlab("backtest", config, "--no-chart")

        assert completed.returncode != 0
        assert "a bot is a folder holding `bot.toml`" in completed.stderr
        assert f"`{config.parent}`" in completed.stderr


class TestFolderWithNoBot:
    def test_the_command_names_the_file_it_expects(self, workspace):
        empty = workspace / "empty"
        empty.mkdir()

        completed = _rtlab("backtest", empty, "--no-chart")

        assert completed.returncode != 0
        assert "bot.toml" in completed.stderr


def _bot(folder: Path, declaration: str) -> Path:
    folder.mkdir(parents=True)
    (folder / "bot.toml").write_text(declaration, encoding="utf-8")
    return folder


def _venue_bot(folder: Path) -> Path:
    declaration = LIVE_BOT.replace(
        'ohlcv_provider = "mock"', 'ohlcv_provider = "ccxt_bitget"'
    )
    return _bot(folder, declaration)


def _rtlab(
    command: str, folder: Path, *options: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603
        [sys.executable, "-m", "robottraderslab", command, str(folder), *options],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
