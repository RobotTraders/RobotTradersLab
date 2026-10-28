import logging
from collections.abc import Iterator
from io import StringIO

import pytest

from robottraderslab.backtester.backtest_logging import (
    logger_levels,
    recorded_log,
    setup_backtest_logging,
)


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    ccxt = logging.getLogger("ccxt")
    handlers, root_level, ccxt_level = root.handlers[:], root.level, ccxt.level
    yield
    root.handlers = handlers
    root.setLevel(root_level)
    ccxt.setLevel(ccxt_level)


def test_uses_custom_path_when_provided(tmp_path):
    custom_path = tmp_path / "custom_backtest.log"

    setup_backtest_logging(
        logfile=custom_path,
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
    )

    assert custom_path.exists()


def test_creates_parent_directories_for_custom_path(tmp_path):
    logfile = tmp_path / "deeply" / "nested" / "path" / "test.log"

    assert not logfile.parent.exists()

    setup_backtest_logging(
        logfile=logfile,
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
    )

    assert logfile.parent.exists()
    assert logfile.exists()


def test_configures_file_handler(tmp_path):
    setup_backtest_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
    )

    (file_handler,) = (
        h for h in logging.root.handlers if isinstance(h, logging.FileHandler)
    )
    assert file_handler.baseFilename == str(tmp_path / "test.log")


def test_configures_console_handler_by_default(tmp_path):
    setup_backtest_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
    )

    console_handler = next(
        (
            h
            for h in logging.root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ),
        None,
    )
    assert console_handler is not None


def test_disables_console_when_requested(tmp_path):
    setup_backtest_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_console=False,
    )

    assert len(logging.root.handlers) == 1
    assert isinstance(logging.root.handlers[0], logging.FileHandler)


def test_logs_to_file(tmp_path):
    logfile = tmp_path / "test.log"
    setup_backtest_logging(
        logfile=logfile,
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
    )

    test_logger = logging.getLogger("test.module")
    test_logger.info("Test backtest message")

    for handler in logging.root.handlers:
        handler.flush()

    log_content = logfile.read_text()
    assert "Test backtest message" in log_content


@pytest.mark.parametrize(
    ("console_level", "expected_level"),
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
    ],
)
def test_respects_console_level(tmp_path, console_level, expected_level):
    setup_backtest_logging(
        logfile=tmp_path / "test.log",
        console_level=console_level,
        file_level="INFO",
        enable_console=True,
    )

    console_handler = next(
        h
        for h in logging.root.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    )
    assert console_handler.level == expected_level


@pytest.mark.parametrize(
    ("file_level", "expected_level"),
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
    ],
)
def test_respects_file_level(tmp_path, file_level, expected_level):
    setup_backtest_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level=file_level,
        enable_console=True,
    )

    file_handler = next(
        h for h in logging.root.handlers if isinstance(h, logging.FileHandler)
    )
    assert file_handler.level == expected_level


def test_does_not_disable_existing_loggers(tmp_path, capsys):
    test_logger = logging.getLogger("test.module")

    setup_backtest_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
    )

    test_logger.info("Test message")

    captured = capsys.readouterr()
    assert "Test message" in captured.out


def test_sets_ccxt_logger_to_warning(tmp_path, capsys):
    setup_backtest_logging(
        logfile=tmp_path / "test.log",
        console_level="DEBUG",
        file_level="INFO",
        enable_console=True,
    )

    ccxt_logger = logging.getLogger("ccxt")
    ccxt_logger.debug("Should not appear")
    ccxt_logger.warning("Should appear")

    captured = capsys.readouterr()
    assert "Should not appear" not in captured.out
    assert "Should appear" in captured.out


def test_accepts_empty_custom_handlers_dict(tmp_path):
    setup_backtest_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        custom_handlers={},
    )

    assert len(logging.root.handlers) == 2


def test_adds_custom_handler_to_root_handlers(tmp_path):
    custom_handlers = {
        "custom": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        }
    }

    setup_backtest_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        custom_handlers=custom_handlers,
    )

    assert len(logging.root.handlers) == 3


def test_adds_multiple_custom_handlers(tmp_path):
    custom_handlers = {
        "custom1": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        },
        "custom2": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "default",
        },
    }

    setup_backtest_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        custom_handlers=custom_handlers,
    )

    assert len(logging.root.handlers) == 4


def test_custom_handlers_work_without_console(tmp_path):
    custom_handlers = {
        "custom": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        }
    }

    setup_backtest_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_console=False,
        custom_handlers=custom_handlers,
    )

    assert len(logging.root.handlers) == 2


def test_custom_handlers_work_without_file(tmp_path):
    custom_handlers = {
        "custom": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        }
    }

    setup_backtest_logging(
        logfile=None,
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        custom_handlers=custom_handlers,
    )

    assert len(logging.root.handlers) == 2


@pytest.mark.usefixtures("restore_logging")
def test_a_recorded_log_holds_what_the_caller_logging_would_drop():
    logging.getLogger().setLevel(logging.WARNING)

    with recorded_log("INFO") as log:
        logging.getLogger("test.module").info("Recorded backtest message")

    assert "INFO: Recorded backtest message" in log.getvalue()


@pytest.mark.usefixtures("restore_logging")
def test_a_recorded_log_leaves_out_what_falls_below_its_level():
    with recorded_log("WARNING") as log:
        logging.getLogger("test.module").info("below-level-99")

    assert "below-level-99" not in log.getvalue()


@pytest.mark.usefixtures("restore_logging")
@pytest.mark.parametrize(
    "quieted_logger", ["ccxt", "robottraderslab._core.actions.executor"]
)
def test_a_recorded_log_leaves_out_the_quieted_loggers(quieted_logger):
    with recorded_log("DEBUG") as log:
        logging.getLogger(quieted_logger).info("chatter-99")

    assert "chatter-99" not in log.getvalue()


@pytest.mark.usefixtures("restore_logging")
def test_a_recorded_log_leaves_the_caller_logging_as_it_found_it():
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    handlers_before = root.handlers[:]
    logging.getLogger("ccxt").setLevel(logging.ERROR)

    with recorded_log("DEBUG"):
        pass

    assert root.level == logging.WARNING
    assert root.handlers == handlers_before
    assert logging.getLogger("ccxt").level == logging.ERROR


@pytest.mark.usefixtures("restore_logging")
def test_a_recorded_log_adds_no_line_to_the_caller_handlers():
    root = logging.getLogger()
    caller_handler = logging.StreamHandler(StringIO())
    root.handlers = [caller_handler]
    root.setLevel(logging.WARNING)

    with recorded_log("INFO"):
        logging.getLogger("test.module").info("run-info-99")

    assert "run-info-99" not in caller_handler.stream.getvalue()
    assert caller_handler.level == logging.NOTSET


@pytest.mark.usefixtures("restore_logging")
def test_a_recorded_log_keeps_warnings_on_stderr_for_an_unconfigured_caller(capsys):
    logging.getLogger().handlers = []

    with recorded_log("INFO"):
        logging.getLogger("test.module").warning("venue-warning-99")

    assert "venue-warning-99" in capsys.readouterr().err
    assert logging.getLogger().handlers == []


def test_every_level_is_restored_when_the_block_raises():
    first = logging.getLogger("test.levels.first")
    second = logging.getLogger("test.levels.second")
    first.setLevel(logging.ERROR)
    second.setLevel(logging.DEBUG)

    with pytest.raises(RuntimeError, match="body-failed-99"):
        with logger_levels({first: logging.INFO, second: logging.CRITICAL}):
            raise RuntimeError("body-failed-99")

    assert first.level == logging.ERROR
    assert second.level == logging.DEBUG
