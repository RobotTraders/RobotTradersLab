import asyncio
import logging
import os
import shutil
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from robottraderslab._core import (
    EX_CONFIG,
    EX_DATAERR,
    EX_TEMPFAIL,
    hold_instance_lock,
)
from robottraderslab.exceptions import StrategyCriticalError
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
    config = tmp_path / "bot.toml"
    config.write_text("[strategy]\nstrategy_class = 'dummy'\n")
    return config


def _launch_args(config: Path) -> tuple[str, ...]:
    return (sys.executable, "-m", "robottraderslab", "live", str(config))


def test_launches_a_bot_registered_without_timeframes(
    create_registry, bot_config, mock_subprocess
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')

    failed_bots = run_scheduler(registry_file)

    mock_subprocess.assert_awaited_once_with(
        *_launch_args(bot_config), stderr=asyncio.subprocess.PIPE
    )
    assert failed_bots == 0


def test_launches_a_bot_on_its_candle_boundary(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock
):
    registry_file = create_registry(
        '[[bot]]\nconfig = "bot.toml"\ntimeframes = ["15m"]\n'
    )
    candle_boundary_clock.now.return_value = datetime(2026, 1, 1, 9, 15, tzinfo=UTC)

    run_scheduler(registry_file)

    mock_subprocess.assert_awaited_once()


def test_skips_a_bot_off_its_candle_boundary(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock
):
    registry_file = create_registry(
        '[[bot]]\nconfig = "bot.toml"\ntimeframes = ["15m"]\n'
    )
    candle_boundary_clock.now.return_value = datetime(2026, 1, 1, 9, 7, tzinfo=UTC)

    run_scheduler(registry_file)

    mock_subprocess.assert_not_awaited()


def test_launches_a_bot_when_one_of_its_timeframes_closes(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock
):
    registry_file = create_registry(
        '[[bot]]\nconfig = "bot.toml"\ntimeframes = ["1h", "4h"]\n'
    )
    candle_boundary_clock.now.return_value = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    run_scheduler(registry_file)

    mock_subprocess.assert_awaited_once()


def test_relative_paths_anchor_to_the_registry_file(
    tmp_path, create_registry, mock_subprocess
):
    nested_config = tmp_path / "bots" / "envelope" / "envelope.toml"
    nested_config.parent.mkdir(parents=True)
    nested_config.write_text("[strategy]\nstrategy_class = 'dummy'\n")
    registry_file = create_registry('[[bot]]\nconfig = "bots/envelope/envelope.toml"\n')

    run_scheduler(registry_file)

    mock_subprocess.assert_awaited_once_with(
        *_launch_args(nested_config), stderr=asyncio.subprocess.PIPE
    )


def test_missing_config_is_skipped_and_logged(
    create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry(
        '[[bot]]\nconfig = "ghost.toml"\n\n[[bot]]\nconfig = "bot.toml"\n'
    )

    with caplog.at_level(logging.ERROR):
        run_scheduler(registry_file)

    assert "does not exist" in caplog.text
    mock_subprocess.assert_awaited_once_with(
        *_launch_args(bot_config), stderr=asyncio.subprocess.PIPE
    )


def test_reports_bots_that_exited_with_an_error(
    create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=1)

    with caplog.at_level(logging.ERROR):
        failed_bots = run_scheduler(registry_file)

    assert failed_bots == 1
    assert (
        caplog.records[-1].getMessage()
        == "bot.toml exited 1, paused, next attempt at 09:08 UTC"
    )


def test_a_broken_config_is_paused_without_the_bots_own_message(
    create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)
    mock_subprocess.return_value.communicate = AsyncMock(
        return_value=(None, b"Error: FET/USDT:USDT has no usable contract on Bitget\n")
    )

    with caplog.at_level(logging.ERROR):
        failed_bots = run_scheduler(registry_file)

    assert failed_bots == 1
    assert [record.getMessage() for record in caplog.records] == [
        "bot.toml exited 78, paused, next attempt at 09:08 UTC"
    ]


@pytest.mark.parametrize("exit_code", [EX_DATAERR, 1])
def test_a_bot_that_died_before_its_log_has_its_message_carried(
    exit_code, create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=exit_code)
    mock_subprocess.return_value.communicate = AsyncMock(
        return_value=(None, b"Error: strategy config is broken\n")
    )

    with caplog.at_level(logging.ERROR):
        run_scheduler(registry_file)

    assert caplog.records[-1].getMessage() == (
        f"bot.toml exited {exit_code}, paused, next attempt at 09:08 UTC: "
        "Error: strategy config is broken"
    )


def test_a_crash_report_survives_undecodable_stderr_bytes(
    create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=1)
    mock_subprocess.return_value.communicate = AsyncMock(
        return_value=(None, b"\xffboom\n")
    )

    with caplog.at_level(logging.ERROR):
        run_scheduler(registry_file)

    assert (
        caplog.records[-1].getMessage()
        == "bot.toml exited 1, paused, next attempt at 09:08 UTC: �boom"
    )


def test_a_clean_exit_is_not_logged_as_an_error(
    create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
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


def test_registry_entry_without_a_config_is_skipped_and_logged(
    create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry(
        '[[bot]]\ntimeframes = ["1h"]\n\n[[bot]]\nconfig = "bot.toml"\n'
    )

    with caplog.at_level(logging.ERROR):
        run_scheduler(registry_file)

    assert "has no `config`" in caplog.text
    mock_subprocess.assert_awaited_once_with(
        *_launch_args(bot_config), stderr=asyncio.subprocess.PIPE
    )


def test_malformed_registry_stops_the_scheduler(create_registry):
    registry_file = create_registry("[[bots]\nconfig = 'bot.toml'\n")

    with pytest.raises(StrategyCriticalError, match="not valid TOML"):
        run_scheduler(registry_file)


def test_missing_registry_stops_the_scheduler(tmp_path):
    with pytest.raises(StrategyCriticalError, match="there is no registry file at"):
        run_scheduler(tmp_path / "absent.toml")


def test_nothing_due_is_not_reported_at_info(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock, caplog
):
    registry_file = create_registry(
        '[[bot]]\nconfig = "bot.toml"\ntimeframes = ["15m"]\n'
    )
    candle_boundary_clock.now.return_value = datetime(2026, 1, 1, 9, 7, tzinfo=UTC)

    with caplog.at_level(logging.INFO):
        run_scheduler(registry_file)

    assert "nothing to launch" not in caplog.text


def test_an_exchange_setting_reaches_the_launched_bots(
    create_registry, bot_config, mock_subprocess, monkeypatch
):
    monkeypatch.delenv("BITGET_MARKET_PACE_FRACTION", raising=False)
    registry_file = create_registry(
        '[[bot]]\nconfig = "bot.toml"\n[exchanges.bitget]\nmarket_pace_fraction = 0.5\n'
    )

    run_scheduler(registry_file)

    assert os.environ["BITGET_MARKET_PACE_FRACTION"] == "0.5"


def test_a_notifier_setting_reaches_the_launched_bots(
    create_registry, bot_config, mock_subprocess, monkeypatch
):
    monkeypatch.delenv("DISCORD_YELLOW", raising=False)
    monkeypatch.delenv("DISCORD_LOG", raising=False)
    registry_file = create_registry(
        '[[bot]]\nconfig = "bot.toml"\n'
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
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')

    run_scheduler(registry_file)

    assert os.environ["BITGET_MARKET_PACE_FRACTION"] == "0.5"


def test_a_bot_that_cannot_be_launched_leaves_the_others_running(
    create_registry, tmp_path, mock_subprocess, caplog
):
    for name in ("first.toml", "second.toml"):
        (tmp_path / name).write_text("[strategy]\nstrategy_class = 'dummy'\n")
    registry_file = create_registry(
        '[[bot]]\nconfig = "first.toml"\n\n[[bot]]\nconfig = "second.toml"\n'
    )
    launched = Mock(spec=asyncio.subprocess.Process)
    launched.wait = AsyncMock(return_value=0)
    launched.communicate = AsyncMock(return_value=(None, b""))
    mock_subprocess.side_effect = [OSError("cannot spawn"), launched]

    with caplog.at_level(logging.ERROR):
        failed_bots = run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2
    assert failed_bots == 1
    assert "Cannot launch first.toml" in caplog.text


def test_a_run_while_the_previous_one_still_holds_the_registry(
    create_registry, bot_config, mock_subprocess, caplog, tmp_path
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')

    with hold_instance_lock(registry_file), caplog.at_level(logging.WARNING):
        failed_bots = run_scheduler(registry_file)

    mock_subprocess.assert_not_awaited()
    assert failed_bots == 0
    assert [record.getMessage() for record in caplog.records] == [
        "Previous run still going, launching nothing from scheduler.toml"
    ]
    assert (tmp_path / ".rtlab" / "scheduler.lock").is_file()


def test_two_consecutive_runs_both_launch(create_registry, bot_config, mock_subprocess):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')

    run_scheduler(registry_file)
    run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2


def test_the_run_after_a_crash_launches(create_registry, bot_config, mock_subprocess):
    registry_file = create_registry(
        'exchanges = "broken"\n[[bot]]\nconfig = "bot.toml"\n'
    )

    with pytest.raises(AttributeError):
        run_scheduler(registry_file)
    create_registry('[[bot]]\nconfig = "bot.toml"\n')
    run_scheduler(registry_file)

    mock_subprocess.assert_awaited_once()


def test_a_bot_outrunning_the_minute(
    create_registry, bot_config, mock_subprocess, bot_clock, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    bot_clock.monotonic.side_effect = [0.0, 70.0]

    with caplog.at_level(logging.WARNING):
        run_scheduler(registry_file)

    assert (
        caplog.records[-1].getMessage()
        == "bot.toml took 70 seconds, longer than the minute between runs"
    )


def test_a_bot_within_the_minute(
    create_registry, bot_config, mock_subprocess, bot_clock, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    bot_clock.monotonic.side_effect = [0.0, 60.0]

    with caplog.at_level(logging.WARNING):
        run_scheduler(registry_file)

    assert caplog.text == ""


def test_a_bot_yielding_to_a_running_instance_is_not_a_failure(
    create_registry, bot_config, mock_subprocess, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_TEMPFAIL)

    with caplog.at_level(logging.ERROR):
        failed_bots = run_scheduler(registry_file)

    assert failed_bots == 0
    assert caplog.text == ""


def test_a_temporary_failure_is_launched_again_on_the_next_run(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_TEMPFAIL)
    mock_subprocess.return_value.communicate = AsyncMock(
        return_value=(None, b"Error: the venue timed out\n")
    )

    with caplog.at_level(logging.WARNING):
        run_scheduler(registry_file)
        candle_boundary_clock.now.return_value = FIRST_RUN + timedelta(minutes=1)
        run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2
    assert caplog.text.count("bot.toml exited 75, retrying every run\n") == 1
    assert "the venue timed out" not in caplog.text


def test_a_paused_bot_is_launched_again_after_one_two_four_then_every_five_minutes(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)

    launches_after_each_run = []
    for minute in range(20):
        candle_boundary_clock.now.return_value = FIRST_RUN + timedelta(minutes=minute)
        run_scheduler(registry_file)
        launches_after_each_run.append(mock_subprocess.await_count)

    assert launches_after_each_run == LAUNCHES_AFTER_EACH_RUN


def test_a_pause_from_a_failure_inside_a_minute_ends_on_the_next_run(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)
    candle_boundary_clock.now.return_value = FIRST_RUN.replace(second=30)

    run_scheduler(registry_file)
    candle_boundary_clock.now.return_value = FIRST_RUN + timedelta(minutes=1)
    run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2


def test_editing_a_paused_bots_config_launches_it_again(
    create_registry, bot_config, mock_subprocess
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)

    run_scheduler(registry_file)
    run_scheduler(registry_file)
    os.utime(bot_config, ns=(EDITED_AT_NS, EDITED_AT_NS))
    run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2


def test_the_same_failure_is_reported_once_over_consecutive_runs(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)

    with caplog.at_level(logging.ERROR):
        for minute in (0, 1, 3):
            candle_boundary_clock.now.return_value = FIRST_RUN + timedelta(
                minutes=minute
            )
            run_scheduler(registry_file)

    assert mock_subprocess.await_count == 3
    assert [record.getMessage() for record in caplog.records] == [
        "bot.toml exited 78, paused, next attempt at 09:08 UTC"
    ]


def test_a_bot_running_again_is_reported_once(
    create_registry, bot_config, mock_subprocess, candle_boundary_clock, caplog
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)
    run_scheduler(registry_file)
    mock_subprocess.return_value.wait = AsyncMock(return_value=0)
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        for minute in (1, 2):
            candle_boundary_clock.now.return_value = FIRST_RUN + timedelta(
                minutes=minute
            )
            run_scheduler(registry_file)

    assert [record.getMessage() for record in caplog.records] == [
        "bot.toml is running again"
    ]


def test_deleting_the_logs_leaves_a_pause_standing(
    create_registry, bot_config, mock_subprocess, tmp_path
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
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
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
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
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)

    run_scheduler(registry_file)
    bot_config.unlink()
    run_scheduler(registry_file)

    memory = (tmp_path / ".rtlab" / "scheduler.launches.json").read_text()

    assert "bot.toml" not in memory


def test_a_deleted_bot_leaves_no_lock_file_behind(
    create_registry, bot_config, mock_subprocess, tmp_path
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')
    with hold_instance_lock(bot_config):
        pass
    bot_config.unlink()

    run_scheduler(registry_file)

    assert not (tmp_path / ".rtlab" / "bot.lock").exists()


def test_a_deleted_bot_loses_its_lock_under_its_own_tool_directory(
    create_registry, mock_subprocess, tmp_path
):
    folder = tmp_path / "pilot"
    (folder / ".rtlab").mkdir(parents=True)
    gone = folder / "gone.toml"
    gone.write_text("[strategy]\n")
    staying = folder / "staying.toml"
    staying.write_text("[strategy]\n")
    registry_file = create_registry(
        '[[bot]]\nconfig = "pilot/gone.toml"\n[[bot]]\nconfig = "pilot/staying.toml"\n'
    )
    with hold_instance_lock(gone):
        pass
    gone.unlink()

    run_scheduler(registry_file)

    assert not (folder / ".rtlab" / "gone.lock").exists()


def test_a_bot_still_there_keeps_its_pause_when_another_is_deleted(
    create_registry, mock_subprocess, tmp_path
):
    deleted = tmp_path / "deleted.toml"
    deleted.write_text("[strategy]\n")
    staying = tmp_path / "staying.toml"
    staying.write_text("[strategy]\n")
    registry_file = create_registry(
        '[[bot]]\nconfig = "deleted.toml"\n[[bot]]\nconfig = "staying.toml"\n'
    )
    mock_subprocess.return_value.wait = AsyncMock(return_value=EX_CONFIG)

    run_scheduler(registry_file)
    deleted.unlink()
    run_scheduler(registry_file)

    assert mock_subprocess.await_count == 2
