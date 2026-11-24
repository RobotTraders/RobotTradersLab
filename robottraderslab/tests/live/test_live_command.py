import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import pytest
from live_bot_helpers import (
    BOT_FOLDER,
    TWO,
    cycle,
    entry_symbols,
    logged,
    save_bot,
)
from live_cycle_helpers import BTC, ETH

from robottraderslab._internal import EX_TEMPFAIL, hold_instance_lock
from robottraderslab.exceptions import BotCriticalError, ExchangeCriticalError

LOGGING_HANDLER_GROUP = "robot_traders_lab.logging_handlers"
DEFAULT_RETENTION_DAYS = 60


def test_the_log_falls_back_to_the_bot_folder(bot_folder):
    save_bot(bot_folder, *TWO)

    cycle(bot_folder)

    assert logged(bot_folder, "INFO") != []
    assert _retention_days() == DEFAULT_RETENTION_DAYS
    assert _console_handlers() == []


def test_the_given_logfile_wins_over_the_config_file(bot_folder, tmp_path):
    save_bot(bot_folder, *TWO)
    logfile = tmp_path / "elsewhere" / "my-logfile.log"

    cycle(bot_folder, logfile_override=logfile)

    assert logfile.is_file()
    assert not (bot_folder / "logs" / "live.log").exists()


def test_retention_comes_from_the_config_file(bot_folder):
    save_bot(bot_folder, *TWO, tail="[live.logging]\nretention_days = 7\n")

    cycle(bot_folder)

    assert _retention_days() == 7


def test_the_given_retention_wins_over_the_config_file(bot_folder):
    save_bot(bot_folder, *TWO, tail="[live.logging]\nretention_days = 7\n")

    cycle(bot_folder, logfiles_retention_days=3)

    assert _retention_days() == 3


def test_a_folder_holding_no_config_file(bot_folder, venue):
    with pytest.raises(BotCriticalError, match="There is no `bot.toml`"):
        cycle(bot_folder)

    venue.place_market_order.assert_not_called()


def test_a_notifier_that_cannot_load_does_not_stop_the_cycle(bot_folder, venue):
    save_bot(bot_folder, *TWO, tail="[live.notifier.nosuchplugin.log]\n")

    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert entry_symbols(venue) == {str(BTC), str(ETH)}
    assert (
        "Running without the `nosuchplugin` notifier" in logged(bot_folder, "ERROR")[0]
    )


def test_a_log_occasion_secret_is_resolved_before_logging_setup(
    bot_folder, make_registered_adapter
):
    received: list[dict[str, str]] = []

    def make_handler(**settings: str) -> logging.Handler:
        received.append(settings)
        return logging.NullHandler()

    make_registered_adapter(make_handler, "recorder", group=LOGGING_HANDLER_GROUP)
    with (bot_folder / "creds.toml").open("a") as secrets:
        secrets.write(
            '\n[[secrets]]\nname = "recorder_main"\nwebhook_url = "https://from-secret"\n'
        )
    save_bot(
        bot_folder,
        *TWO,
        tail='[live.notifier.recorder.log]\nsecret_name = "recorder_main"\n',
    )

    cycle(bot_folder)

    assert received == [{"webhook_url": "https://from-secret"}]


def test_a_log_occasion_secret_the_file_lacks_leaves_the_notifier_out(
    bot_folder, venue
):
    save_bot(
        bot_folder,
        *TWO,
        tail='[live.notifier.recorder.log]\nsecret_name = "absent"\n',
    )

    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert entry_symbols(venue) == {str(BTC), str(ETH)}
    assert "Running without the `recorder` notifier" in logged(bot_folder, "ERROR")[0]
    assert "absent" in logged(bot_folder, "ERROR")[0]


def test_a_config_already_trading_in_another_process(bot_folder, venue):
    save_bot(bot_folder, *TWO)

    with hold_instance_lock(bot_folder / "bot.toml"):
        exit_code = cycle(bot_folder)

    assert exit_code == EX_TEMPFAIL
    venue.place_market_order.assert_not_called()
    assert f"{BOT_FOLDER} is already trading in another process" in "".join(
        logged(bot_folder, "WARNING")
    )


def test_a_cycle_keeps_its_files_in_the_bot_folder_and_nothing_in_the_workspace(
    bot_folder, tmp_path, names_in_tool_directory
):
    (tmp_path / ".rtlab").mkdir()
    save_bot(bot_folder, *TWO)

    cycle(bot_folder)

    assert (bot_folder / ".rtlab" / "bot.lock").is_file()
    assert (bot_folder / ".rtlab" / "bot.working.toml").is_file()
    assert names_in_tool_directory(tmp_path) == []


def test_the_next_cycle_after_a_crash(bot_folder, venue):
    save_bot(bot_folder, *TWO)
    venue.get_open_positions.side_effect = [
        ExchangeCriticalError("the venue rejected us"),
        ExchangeCriticalError("the venue rejected us"),
        {},
        {},
    ]

    with pytest.raises(ExchangeCriticalError):
        cycle(bot_folder)
    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert entry_symbols(venue) == {str(BTC), str(ETH)}


def test_a_folder_given_by_a_relative_path(bot_folder, monkeypatch):
    save_bot(bot_folder, *TWO)
    monkeypatch.chdir(bot_folder.parent)

    exit_code = cycle(Path(BOT_FOLDER))

    assert exit_code == 0


def _retention_days() -> int:
    """The days of logs the bot's log file keeps."""
    (handler,) = (
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, TimedRotatingFileHandler)
    )
    return handler.backupCount


def _console_handlers() -> list[logging.Handler]:
    """The handlers writing the log to a console."""
    return [
        handler
        for handler in logging.getLogger().handlers
        if type(handler) is logging.StreamHandler
    ]
