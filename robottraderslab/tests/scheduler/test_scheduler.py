import asyncio
import logging
import os
import re
import shutil
import sys
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from robottraderslab._internal import (
    EX_CONFIG,
    EX_DATAERR,
    EX_TEMPFAIL,
    hold_scheduler_lock,
)
from robottraderslab.exceptions import BotCriticalError
from robottraderslab.live import working_copy_file
from robottraderslab.scheduler import run_scheduler

FIRST_RUN = datetime(2026, 1, 1, 9, 7, tzinfo=UTC)
EDITED_AT_NS = 1_700_000_000_000_000_000
LAUNCHES_AFTER_EACH_RUN = [1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6]


@pytest.fixture(autouse=True)
def candle_boundary_clock() -> Iterator[Mock]:
    with patch("robottraderslab.scheduler.scheduler.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2026, 1, 1, 9, 7, tzinfo=UTC)
        yield mock_datetime


@pytest.fixture
def mock_subprocess() -> Iterator[AsyncMock]:
    process = Mock(spec=asyncio.subprocess.Process)
    process.wait = AsyncMock(return_value=0)
    process.communicate = AsyncMock(return_value=(None, b""))
    with patch(
        "robottraderslab.scheduler.scheduler.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
        return_value=process,
    ) as mock_exec:
        yield mock_exec


@pytest.fixture
def bot_clock() -> Iterator[Mock]:
    with patch("robottraderslab.scheduler.scheduler.time") as mock_time:
        yield mock_time


@pytest.fixture
def bot_config(tmp_path: Path) -> Path:
    config = tmp_path / "pilot" / "bot.toml"
    config.parent.mkdir()
    config.write_text("[strategy]\nstrategy_class = 'dummy'\n")
    return config


@pytest.fixture
def copy_only_bot(
    bot_config: Path, keep_only_working_copy: Callable[[Path], Path]
) -> Path:
    keep_only_working_copy(bot_config)
    return bot_config


@pytest.fixture(params=["bot_config", "copy_only_bot"], ids=["bot file", "copy only"])
def pilot_bot(request: pytest.FixtureRequest) -> Path:
    bot_file: Path = request.getfixturevalue(request.param)
    return bot_file


def _launch_args(config: Path) -> tuple[str, ...]:
    return (sys.executable, "-m", "robottraderslab", "live", str(config.parent))


def test_launches_a_bot_registered_without_timeframes(
    create_registry, bot_config, mock_subprocess
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')

    failed_bots = run_scheduler(registry_file)

    _assert_launched_alone(mock_subprocess, bot_config)
    assert failed_bots == 0


def test_launches_a_bot_on_its_candle_boundary(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\ntimeframes = ["15m"]\n')
    candle_boundary_clock.now.return_value = datetime(2026, 1, 1, 9, 15, tzinfo=UTC)

    run_scheduler(registry_file)

    mock_subprocess.assert_awaited_once()


def test_skips_a_bot_off_its_candle_boundary(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\ntimeframes = ["15m"]\n')
    candle_boundary_clock.now.return_value = datetime(2026, 1, 1, 9, 7, tzinfo=UTC)

    run_scheduler(registry_file)

    mock_subprocess.assert_not_awaited()


def test_launches_a_bot_when_one_of_its_timeframes_closes(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock
):
    registry_file = create_registry(
        '[[bot]]\nconfig = "pilot"\ntimeframes = ["1h", "4h"]\n'
    )
    candle_boundary_clock.now.return_value = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    run_scheduler(registry_file)

    mock_subprocess.assert_awaited_once()


def test_a_nested_bot_folder_is_anchored_to_the_registry_file(
    tmp_path, create_registry, mock_subprocess
):
    nested_config = tmp_path / "trend" / "first-bot" / "bot.toml"
    nested_config.parent.mkdir(parents=True)
    nested_config.write_text("[strategy]\nstrategy_class = 'dummy'\n")
    registry_file = create_registry('[[bot]]\nconfig = "trend/first-bot"\n')

    run_scheduler(registry_file)

    _assert_launched_alone(mock_subprocess, nested_config)


def test_an_entry_naming_a_toml_file_that_is_gone_stops_the_scheduler(
    create_registry, bot_config, mock_subprocess
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot/impulse-pilot.toml"\n')

    with pytest.raises(BotCriticalError, match="a bot is a folder") as refusal:
        run_scheduler(registry_file)

    assert f"`{bot_config.parent}`" in str(refusal.value)
    mock_subprocess.assert_not_awaited()


def test_an_entry_naming_a_toml_file_stops_the_scheduler_naming_the_folder(
    create_registry, bot_config, mock_subprocess
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot/bot.toml"\n')

    with pytest.raises(BotCriticalError, match="a bot is a folder") as refusal:
        run_scheduler(registry_file)

    assert f"`{bot_config.parent}`" in str(refusal.value)
    mock_subprocess.assert_not_awaited()


def test_a_folder_without_a_bot_file_is_skipped_and_logged(
    create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry(
        '[[bot]]\nconfig = "ghost"\n\n[[bot]]\nconfig = "pilot"\n'
    )

    with caplog.at_level(logging.ERROR):
        run_scheduler(registry_file)

    assert [record.getMessage() for record in caplog.records] == [
        f"There is no `bot.toml` in `{bot_config.parent.with_name('ghost')}`, "
        "skipping it"
    ]
    _assert_launched_alone(mock_subprocess, bot_config)


def test_reports_bots_that_exited_with_an_error(
    create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=1)

    with caplog.at_level(logging.ERROR):
        failed_bots = run_scheduler(registry_file)

    assert failed_bots == 1
    assert (
        caplog.records[-1].getMessage()
        == "pilot exited 1, paused, next attempt at 09:08 UTC"
    )


def test_a_broken_config_is_paused_without_the_bots_own_message(
    create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)
    mock_subprocess.return_value.communicate = AsyncMock(
        return_value=(None, b"Error: FET/USDT:USDT has no usable contract on Bitget\n")
    )

    with caplog.at_level(logging.ERROR):
        failed_bots = run_scheduler(registry_file)

    assert failed_bots == 1
    assert [record.getMessage() for record in caplog.records] == [
        "pilot exited 78, paused, next attempt at 09:08 UTC"
    ]


@pytest.mark.parametrize("exit_code", [EX_DATAERR, 1])
def test_a_bot_that_died_before_its_log_has_its_message_carried(
    exit_code, create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=exit_code)
    mock_subprocess.return_value.communicate = AsyncMock(
        return_value=(None, b"Error: strategy config is broken\n")
    )

    with caplog.at_level(logging.ERROR):
        run_scheduler(registry_file)

    assert caplog.records[-1].getMessage() == (
        f"pilot exited {exit_code}, paused, next attempt at 09:08 UTC: "
        "Error: strategy config is broken"
    )


def test_a_crash_report_survives_undecodable_stderr_bytes(
    create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=1)
    mock_subprocess.return_value.communicate = AsyncMock(
        return_value=(None, b"\xffboom\n")
    )

    with caplog.at_level(logging.ERROR):
        run_scheduler(registry_file)

    assert (
        caplog.records[-1].getMessage()
        == "pilot exited 1, paused, next attempt at 09:08 UTC: �boom"
    )


def test_a_clean_exit_is_not_logged_as_an_error(
    create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.communicate = AsyncMock(
        return_value=(None, b"a benign warning\n")
    )

    with caplog.at_level(logging.ERROR):
        run_scheduler(registry_file)

    assert caplog.text == ""


def test_empty_registry_launches_nothing(create_registry, mock_subprocess, caplog):
    registry_file = create_registry("")

    with caplog.at_level(logging.WARNING):
        failed_bots = run_scheduler(registry_file)

    assert "No bots registered" in caplog.text
    mock_subprocess.assert_not_awaited()
    assert failed_bots == 0


def test_registry_entry_without_a_bot_is_skipped_and_logged(
    create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry(
        '[[bot]]\ntimeframes = ["1h"]\n\n[[bot]]\nconfig = "pilot"\n'
    )

    with caplog.at_level(logging.ERROR):
        run_scheduler(registry_file)

    assert "has no `config`" in caplog.text
    _assert_launched_alone(mock_subprocess, bot_config)


def test_malformed_registry_stops_the_scheduler(create_registry):
    registry_file = create_registry("[[bots]\nbot = 'pilot'\n")

    with pytest.raises(BotCriticalError, match="not valid TOML"):
        run_scheduler(registry_file)


def test_a_registry_naming_two_bots_on_one_symbol_of_one_account_stops_the_scheduler(
    create_registry, two_bots_on_one_symbol
):
    registry_file = _register(create_registry, two_bots_on_one_symbol)

    with pytest.raises(BotCriticalError, match="BTC/USDT:USDT"):
        run_scheduler(registry_file)


def test_missing_registry_stops_the_scheduler(tmp_path):
    with pytest.raises(BotCriticalError, match="there is no registry file at"):
        run_scheduler(tmp_path / "absent.toml")


def test_nothing_due_is_not_reported_at_info(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\ntimeframes = ["15m"]\n')
    candle_boundary_clock.now.return_value = datetime(2026, 1, 1, 9, 7, tzinfo=UTC)

    with caplog.at_level(logging.INFO):
        run_scheduler(registry_file)

    assert "nothing to launch" not in caplog.text


def test_an_exchange_setting_reaches_the_launched_bots(
    create_registry, bot_config, mock_subprocess, monkeypatch
):
    monkeypatch.delenv("BITGET_MARKET_PACE_FRACTION", raising=False)
    registry_file = create_registry(
        '[[bot]]\nconfig = "pilot"\n[exchanges.bitget]\nmarket_pace_fraction = 0.5\n'
    )

    run_scheduler(registry_file)

    assert os.environ["BITGET_MARKET_PACE_FRACTION"] == "0.5"


def test_a_notifier_setting_reaches_the_launched_bots(
    create_registry, bot_config, mock_subprocess, monkeypatch
):
    monkeypatch.delenv("DISCORD_YELLOW", raising=False)
    monkeypatch.delenv("DISCORD_LOG", raising=False)
    registry_file = create_registry(
        '[[bot]]\nconfig = "pilot"\n'
        "[notifier.discord]\nyellow = 0xF1C40F\n"
        '[notifier.discord.log]\nwebhook_url = "https://discord.test/hook"\n'
    )

    run_scheduler(registry_file)

    assert os.environ["DISCORD_YELLOW"] == "15844367"
    assert "DISCORD_LOG" not in os.environ


def test_a_registry_without_exchange_settings_changes_nothing(
    create_registry, bot_config, mock_subprocess, monkeypatch
):
    monkeypatch.setenv("BITGET_MARKET_PACE_FRACTION", "0.5")
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')

    run_scheduler(registry_file)

    assert os.environ["BITGET_MARKET_PACE_FRACTION"] == "0.5"


def test_a_bot_that_cannot_be_launched_leaves_the_others_running(
    create_registry, tmp_path, mock_subprocess, caplog
):
    for name in ("first", "second"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "bot.toml").write_text(
            "[strategy]\nstrategy_class = 'dummy'\n"
        )
    registry_file = create_registry(
        '[[bot]]\nconfig = "first"\n\n[[bot]]\nconfig = "second"\n'
    )
    launched = Mock(spec=asyncio.subprocess.Process)
    launched.wait = AsyncMock(return_value=0)
    launched.communicate = AsyncMock(return_value=(None, b""))
    mock_subprocess.side_effect = [OSError("cannot spawn"), launched]

    with caplog.at_level(logging.ERROR):
        failed_bots = run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2
    assert failed_bots == 1
    assert "Cannot launch first" in caplog.text


def test_a_run_while_the_previous_one_still_holds_the_registry(
    create_registry, bot_config, mock_subprocess, caplog, tmp_path
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')

    with hold_scheduler_lock(registry_file), caplog.at_level(logging.WARNING):
        failed_bots = run_scheduler(registry_file)

    mock_subprocess.assert_not_awaited()
    assert failed_bots == 0
    assert [record.getMessage() for record in caplog.records] == [
        "Previous run still going, launching nothing from scheduler.toml"
    ]
    assert (tmp_path / ".rtlab" / "scheduler.lock").is_file()


def test_two_consecutive_runs_both_launch(create_registry, bot_config, mock_subprocess):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')

    run_scheduler(registry_file)
    run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2


def test_the_run_after_a_crash_launches(create_registry, bot_config, mock_subprocess):
    registry_file = create_registry('exchanges = "broken"\n[[bot]]\nconfig = "pilot"\n')

    with pytest.raises(AttributeError):
        run_scheduler(registry_file)
    create_registry('[[bot]]\nconfig = "pilot"\n')
    run_scheduler(registry_file)

    mock_subprocess.assert_awaited_once()


def test_a_bot_outrunning_the_minute(
    create_registry, bot_config, mock_subprocess, bot_clock, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    bot_clock.monotonic.side_effect = [0.0, 70.0]

    with caplog.at_level(logging.WARNING):
        run_scheduler(registry_file)

    assert (
        caplog.records[-1].getMessage()
        == "pilot took 70 seconds, longer than the minute between runs"
    )


def test_a_bot_within_the_minute(
    create_registry, bot_config, mock_subprocess, bot_clock, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    bot_clock.monotonic.side_effect = [0.0, 60.0]

    with caplog.at_level(logging.WARNING):
        run_scheduler(registry_file)

    assert caplog.text == ""


def test_a_bot_yielding_to_a_running_instance_is_not_a_failure(
    create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_TEMPFAIL)

    with caplog.at_level(logging.ERROR):
        failed_bots = run_scheduler(registry_file)

    assert failed_bots == 0
    assert caplog.text == ""


def test_a_temporary_failure_is_launched_again_on_the_next_run(
    create_registry, pilot_bot, mock_subprocess, candle_boundary_clock, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_TEMPFAIL)
    mock_subprocess.return_value.communicate = AsyncMock(
        return_value=(None, b"Error: the venue timed out\n")
    )

    with caplog.at_level(logging.WARNING):
        run_scheduler(registry_file)
        candle_boundary_clock.now.return_value = FIRST_RUN + timedelta(minutes=1)
        run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2
    assert caplog.text.count("pilot exited 75, retrying every run\n") == 1
    assert "the venue timed out" not in caplog.text


def test_a_paused_bot_is_launched_again_after_one_two_four_then_every_five_minutes(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)

    launches_after_each_run = []
    for minute in range(20):
        _run_scheduler_at(
            registry_file, candle_boundary_clock, FIRST_RUN + timedelta(minutes=minute)
        )
        launches_after_each_run.append(mock_subprocess.await_count)

    assert launches_after_each_run == LAUNCHES_AFTER_EACH_RUN


def test_a_pause_from_a_failure_inside_a_minute_ends_on_the_next_run(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)
    candle_boundary_clock.now.return_value = FIRST_RUN.replace(second=30)

    run_scheduler(registry_file)
    candle_boundary_clock.now.return_value = FIRST_RUN + timedelta(minutes=1)
    run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2


def test_editing_a_paused_bots_config_launches_it_again(
    create_registry, bot_config, mock_subprocess
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)

    run_scheduler(registry_file)
    run_scheduler(registry_file)
    os.utime(bot_config, ns=(EDITED_AT_NS, EDITED_AT_NS))
    run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2


def test_the_same_failure_is_reported_once_over_consecutive_runs(
    create_registry, pilot_bot, mock_subprocess, candle_boundary_clock, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)

    with caplog.at_level(logging.ERROR):
        for minute in (0, 1, 3):
            _run_scheduler_at(
                registry_file,
                candle_boundary_clock,
                FIRST_RUN + timedelta(minutes=minute),
            )

    assert mock_subprocess.await_count == 3
    assert [record.getMessage() for record in caplog.records] == [
        "pilot exited 78, paused, next attempt at 09:08 UTC"
    ]


def test_a_bot_running_again_is_reported_once(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)
    run_scheduler(registry_file)
    mock_subprocess.return_value.wait = AsyncMock(return_value=0)
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        for minute in (1, 2):
            _run_scheduler_at(
                registry_file,
                candle_boundary_clock,
                FIRST_RUN + timedelta(minutes=minute),
            )

    assert [record.getMessage() for record in caplog.records] == [
        "pilot is running again"
    ]


def test_deleting_the_logs_leaves_a_pause_standing(
    create_registry, bot_config, mock_subprocess, tmp_path
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "scheduler.log").write_text("one scheduler line")

    run_scheduler(registry_file)
    shutil.rmtree(logs)
    run_scheduler(registry_file)

    assert mock_subprocess.await_count == 1


def test_a_memory_that_cannot_be_read_counts_as_absent(
    create_registry, bot_config, mock_subprocess, tmp_path, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)

    run_scheduler(registry_file)
    (tmp_path / ".rtlab" / "scheduler.launches.json").write_text("not a memory at all")
    with caplog.at_level(logging.WARNING):
        run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2
    assert "Starting with no memory of" in caplog.text


def test_a_deleted_bot_leaves_no_memory_of_its_pause(
    create_registry, bot_config, mock_subprocess, tmp_path
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)

    run_scheduler(registry_file)
    bot_config.unlink()
    run_scheduler(registry_file)

    memory = (tmp_path / ".rtlab" / "scheduler.launches.json").read_text()

    assert "bot.toml" not in memory


def test_a_deleted_bot_folder_leaves_no_file_and_no_pause_behind(
    create_registry,
    bot_config,
    keep_working_copy,
    mock_subprocess,
    tmp_path,
    create_lock_file,
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait.return_value = EX_CONFIG
    copy_file = keep_working_copy(bot_config)
    run_scheduler(registry_file)
    create_lock_file(bot_config)

    shutil.rmtree(bot_config.parent)
    run_scheduler(registry_file)

    remaining = [path.name for path in tmp_path.rglob("*") if "pilot" in path.name]
    assert remaining == []
    assert not copy_file.exists()
    assert list(tmp_path.rglob("bot.working.toml")) == []
    assert "pilot" not in (tmp_path / ".rtlab" / "scheduler.launches.json").read_text()


def test_a_deleted_bot_leaves_no_lock_file_behind(
    create_registry, bot_config, mock_subprocess, tmp_path, create_earlier_lock
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    earlier = create_earlier_lock(tmp_path, bot_config)
    bot_config.unlink()

    run_scheduler(registry_file)

    assert not earlier.exists()


def test_a_bot_whose_config_alone_is_deleted_loses_the_lock_in_its_folder(
    create_registry, bot_config, mock_subprocess, create_lock_file
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    create_lock_file(bot_config)
    bot_config.unlink()

    run_scheduler(registry_file)

    assert not (bot_config.parent / ".rtlab" / "bot.lock").exists()


def test_a_deleted_bot_loses_its_lock_under_its_own_workspace(
    create_registry, mock_subprocess, tmp_path, create_earlier_lock
):
    workspace = tmp_path / "kraken"
    (workspace / ".rtlab").mkdir(parents=True)
    gone = workspace / "gone" / "bot.toml"
    staying = workspace / "staying" / "bot.toml"
    for config in (gone, staying):
        config.parent.mkdir()
        config.write_text("[strategy]\n")
    registry_file = create_registry(
        '[[bot]]\nconfig = "kraken/gone"\n[[bot]]\nconfig = "kraken/staying"\n'
    )
    earlier = create_earlier_lock(workspace, gone)
    gone.unlink()

    run_scheduler(registry_file)

    assert not earlier.exists()


def test_a_run_sweeps_the_bot_locks_of_the_earlier_placement(
    create_registry,
    mock_subprocess,
    tmp_path,
    create_earlier_lock,
    names_in_tool_directory,
):
    registry_file = create_registry('[[bot]]\nconfig = "a"\n[[bot]]\nconfig = "b"\n')
    for name in ("a", "b"):
        config = tmp_path / name / "bot.toml"
        config.parent.mkdir()
        config.write_text("[strategy]\nstrategy_class = 'dummy'\n")
        create_earlier_lock(tmp_path, config)
    mock_subprocess.return_value.wait.return_value = EX_CONFIG

    run_scheduler(registry_file)

    assert names_in_tool_directory(tmp_path) == [
        "scheduler.launches.json",
        "scheduler.lock",
    ]


def test_a_bot_still_there_keeps_its_pause_when_another_is_deleted(
    create_registry, mock_subprocess, tmp_path
):
    deleted = tmp_path / "deleted" / "bot.toml"
    staying = tmp_path / "staying" / "bot.toml"
    for config in (deleted, staying):
        config.parent.mkdir()
        config.write_text("[strategy]\n")
    registry_file = create_registry(
        '[[bot]]\nconfig = "deleted"\n[[bot]]\nconfig = "staying"\n'
    )
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)

    run_scheduler(registry_file)
    deleted.unlink()
    run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2


def test_two_bots_sharing_a_name_and_an_account_are_not_launched(
    create_registry, write_bot, mock_subprocess
):
    write_bot("first", name="grid", account="main")
    write_bot("second", name="grid", account="main")
    registry_file = create_registry(
        '[[bot]]\nconfig = "first/grid"\n[[bot]]\nconfig = "second/grid"\n'
    )

    _assert_registry_refused(
        registry_file, mock_subprocess, "their folder names share an identity"
    )


def test_two_bots_sharing_a_name_on_different_accounts_are_both_launched(
    create_registry, write_bot, mock_subprocess
):
    write_bot("first", name="grid", account="main")
    write_bot("second", name="grid", account="spare")
    registry_file = create_registry(
        '[[bot]]\nconfig = "first/grid"\n[[bot]]\nconfig = "second/grid"\n'
    )

    run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2


SOL = "SOL/USDT:USDT"
REFUSED_VARIABLE = "RTLAB_EDIT_REFUSED"


@pytest.fixture
def registered_bots(
    create_registry: Callable[[str], Path],
    two_bots_with_working_copies: tuple[Path, Path],
) -> tuple[Path, Path, Path]:
    first, second = two_bots_with_working_copies
    return _register(create_registry, [first, second]), first, second


def test_an_edit_naming_another_bots_symbol_launches_both_and_refuses_that_edit_alone(
    registered_bots,
    edit_first_bot,
    mock_subprocess,
    caplog,
    first_symbol,
    second_symbol,
):
    registry_file, first, second = registered_bots
    edit_first_bot(first_symbol, second_symbol)

    with caplog.at_level(logging.ERROR):
        failed_bots = run_scheduler(registry_file)

    assert failed_bots == 0
    assert mock_subprocess.await_count == 2
    sentence = _launch_of(mock_subprocess, first).kwargs["env"][REFUSED_VARIABLE]
    assert sentence == (
        f"`{first}` adds `{second_symbol}` on the account `main` on bitget, "
        "which `scalp` already trades; "
        "remove it from one of the two bots or give each its own account"
    )
    assert "env" not in _launch_of(mock_subprocess, second).kwargs
    assert [record.levelname for record in caplog.records] == ["ERROR"]
    assert sentence in caplog.text


def test_two_bots_edited_to_name_the_same_new_symbol_are_both_launched_with_a_sentence(
    registered_bots,
    edit_first_bot,
    edit_second_bot,
    mock_subprocess,
    caplog,
    first_symbol,
    second_symbol,
):
    registry_file, first, second = registered_bots
    edit_first_bot(first_symbol, SOL)
    edit_second_bot(second_symbol, SOL)

    with caplog.at_level(logging.ERROR):
        run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2
    assert REFUSED_VARIABLE in _launch_of(mock_subprocess, first).kwargs["env"]
    assert REFUSED_VARIABLE in _launch_of(mock_subprocess, second).kwargs["env"]
    assert len(caplog.records) == 2


def test_a_folder_holding_only_a_working_copy_that_conflicts_refuses_the_registry(
    create_registry,
    edit_second_bot,
    keep_only_working_copy,
    keep_working_copy,
    two_bots_on_one_symbol,
    second_symbol,
    mock_subprocess,
):
    first, second = two_bots_on_one_symbol
    keep_only_working_copy(first)
    keep_working_copy(second)
    edit_second_bot(second_symbol)
    registry_file = _register(create_registry, [first, second])

    _assert_registry_refused(registry_file, mock_subprocess, re.escape(f"of `{first}`"))


def test_the_sentence_never_reaches_the_schedulers_own_environment(
    registered_bots,
    edit_first_bot,
    mock_subprocess,
    monkeypatch,
    first_symbol,
    second_symbol,
):
    registry_file, _, _ = registered_bots
    edit_first_bot(first_symbol, second_symbol)
    monkeypatch.delenv(REFUSED_VARIABLE, raising=False)

    run_scheduler(registry_file)

    assert REFUSED_VARIABLE not in os.environ


def test_a_sentence_inherited_from_the_operators_shell_reaches_no_bot(
    registered_bots, mock_subprocess, monkeypatch
):
    registry_file, first, second = registered_bots
    monkeypatch.setenv(REFUSED_VARIABLE, "held back by hand")

    run_scheduler(registry_file)

    assert REFUSED_VARIABLE not in os.environ
    assert "env" not in _launch_of(mock_subprocess, first).kwargs
    assert "env" not in _launch_of(mock_subprocess, second).kwargs


def test_a_refused_edit_is_reported_at_each_launch_of_its_bot_and_not_between(
    registered_bots,
    edit_first_bot,
    mock_subprocess,
    candle_boundary_clock,
    caplog,
    first_symbol,
    second_symbol,
):
    registry_file, first, second = registered_bots
    registry_file.write_text(
        f"[[bot]]\nconfig = '{first.parent.as_posix()}'\ntimeframes = ['4h']\n"
        f"[[bot]]\nconfig = '{second.parent.as_posix()}'\n"
    )
    edit_first_bot(first_symbol, second_symbol)

    with caplog.at_level(logging.ERROR):
        for hour in (9, 12, 16):
            moment = datetime(2026, 1, 1, hour, 0, tzinfo=UTC)
            _run_scheduler_at(registry_file, candle_boundary_clock, moment)

    assert len(caplog.records) == 2
    assert mock_subprocess.await_count == 5


def test_a_folder_holding_only_a_working_copy_is_launched_and_keeps_its_copy_and_lock(
    create_registry, copy_only_bot, mock_subprocess, create_lock_file, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    create_lock_file(copy_only_bot)

    with caplog.at_level(logging.ERROR):
        failed_bots = run_scheduler(registry_file)

    _assert_launched_alone(mock_subprocess, copy_only_bot)
    assert failed_bots == 0
    assert caplog.text == ""
    assert working_copy_file(copy_only_bot).is_file()
    assert (copy_only_bot.parent / ".rtlab" / "bot.lock").is_file()


@pytest.mark.parametrize("unusable", [b"name = [\n", b"\xff\xfe"])
def test_a_folder_with_no_bot_file_and_an_unusable_working_copy_is_skipped_naming_both(
    unusable,
    create_registry,
    copy_only_bot,
    write_unusable_copy,
    mock_subprocess,
    caplog,
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    write_unusable_copy(copy_only_bot, unusable)

    with caplog.at_level(logging.WARNING):
        run_scheduler(registry_file)

    mock_subprocess.assert_not_awaited()
    skip, nothing_registered = caplog.records
    assert skip.levelname == "ERROR"
    assert skip.getMessage() == (
        f"There is no `bot.toml` in `{copy_only_bot.parent}`, and its working "
        f"copy `{working_copy_file(copy_only_bot)}` cannot be used "
        "(unreadable or not valid TOML), skipping it"
    )
    assert nothing_registered.getMessage().startswith("No bots registered in")


def test_a_bot_file_that_comes_back_lifts_the_pause_of_a_copy_only_bot(
    create_registry, copy_only_bot, mock_subprocess
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait.return_value = EX_CONFIG

    run_scheduler(registry_file)
    run_scheduler(registry_file)
    copy_only_bot.write_text("[strategy]\nstrategy_class = 'dummy'\n")
    run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2


def test_a_copy_only_bot_keeps_its_pause_while_its_copy_stands(
    create_registry, copy_only_bot, mock_subprocess, tmp_path
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait.return_value = EX_CONFIG

    run_scheduler(registry_file)
    run_scheduler(registry_file)

    memory_text = (tmp_path / ".rtlab" / "scheduler.launches.json").read_text()
    assert "pilot" in memory_text
    assert mock_subprocess.await_count == 1


def test_a_folder_whose_working_copy_became_unusable_loses_its_pause_memory(
    create_registry, copy_only_bot, write_unusable_copy, mock_subprocess, tmp_path
):
    registry_file = create_registry('[[bot]]\nconfig = "pilot"\n')
    mock_subprocess.return_value.wait.return_value = EX_CONFIG
    run_scheduler(registry_file)

    write_unusable_copy(copy_only_bot)
    run_scheduler(registry_file)

    memory_text = (tmp_path / ".rtlab" / "scheduler.launches.json").read_text()
    assert "pilot" not in memory_text


def _assert_launched_alone(mock_subprocess: AsyncMock, bot_file: Path) -> None:
    mock_subprocess.assert_awaited_once_with(
        *_launch_args(bot_file), stderr=asyncio.subprocess.PIPE
    )


def _run_scheduler_at(registry_file: Path, clock: Mock, moment: datetime) -> int:
    clock.now.return_value = moment
    return run_scheduler(registry_file)


def _assert_registry_refused(
    registry_file: Path, mock_subprocess: AsyncMock, match: str
) -> None:
    with pytest.raises(BotCriticalError, match=match):
        run_scheduler(registry_file)

    mock_subprocess.assert_not_awaited()


def _register(
    create_registry: Callable[[str], Path], bot_files: Iterable[Path]
) -> Path:
    return create_registry(
        "".join(
            f"[[bot]]\nconfig = '{bot_file.parent.as_posix()}'\n"
            for bot_file in bot_files
        )
    )


def _launch_of(mock_subprocess: AsyncMock, bot_file: Path) -> Mock:
    launches = [
        launch
        for launch in mock_subprocess.await_args_list
        if launch.args[-1] == str(bot_file.parent)
    ]
    assert len(launches) == 1
    return launches[0]
