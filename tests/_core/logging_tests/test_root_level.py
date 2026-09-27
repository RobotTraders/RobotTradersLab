import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from robottraderslab._core import LogLevel
from robottraderslab.backtester.backtest_logging import setup_backtest_logging


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    yield
    root_logger.handlers = original_handlers
    root_logger.setLevel(original_level)


def _configure(logfile: Path, file_level: LogLevel, **overrides: Any) -> None:
    setup_backtest_logging(
        logfile=logfile,
        log_name="a-bot",
        console_level=LogLevel.INFO,
        file_level=file_level,
        enable_console=True,
        enable_file=True,
        **overrides,
    )


def test_root_level_follows_the_most_permissive_handler(tmp_path):
    _configure(tmp_path / "backtest.log", LogLevel.WARNING)

    assert logging.getLogger().level == logging.INFO


def test_debug_calls_are_disabled_when_no_handler_consumes_them(tmp_path):
    _configure(tmp_path / "backtest.log", LogLevel.INFO)

    probe_logger = logging.getLogger("robottraderslab.probe")

    assert probe_logger.isEnabledFor(logging.DEBUG) is False


def test_debug_calls_stay_enabled_for_a_debug_handler(tmp_path):
    _configure(tmp_path / "backtest.log", LogLevel.DEBUG)

    probe_logger = logging.getLogger("robottraderslab.probe")

    assert probe_logger.isEnabledFor(logging.DEBUG) is True


def test_custom_handler_without_a_level_keeps_debug(tmp_path):
    _configure(
        tmp_path / "backtest.log",
        LogLevel.INFO,
        custom_handlers={"probe": {"class": "logging.NullHandler"}},
    )

    assert logging.getLogger().level == logging.DEBUG


def test_custom_handler_with_a_numeric_level(tmp_path):
    _configure(
        tmp_path / "backtest.log",
        LogLevel.INFO,
        custom_handlers={"probe": {"class": "logging.NullHandler", "level": 30}},
    )

    assert logging.getLogger().level == logging.INFO
