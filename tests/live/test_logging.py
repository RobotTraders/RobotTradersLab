import logging
import logging.handlers
from pathlib import Path

import pytest

from robottraderslab.live.logging import setup_live_logging


def test_names_the_default_file_after_the_bot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    logfile = setup_live_logging(
        logfile=None,
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
    )

    assert logfile.exists()
    assert logfile.parent.name == "logs"
    assert logfile.name == "a-bot.log"


def test_creates_log_under_config_dir_when_provided(tmp_path: Path):
    config_dir = tmp_path / "workspace" / "my_strategy"
    config_dir.mkdir(parents=True)

    logfile = setup_live_logging(
        logfile=None,
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
        config_dir=config_dir,
    )

    assert logfile.exists()
    assert logfile.parent == config_dir / "logs"
    assert logfile.name == "a-bot.log"


def test_bots_sharing_a_directory_keep_their_own_log(tmp_path: Path):
    config_dir = tmp_path / "pilot"
    config_dir.mkdir()

    impulse_log = setup_live_logging(
        logfile=None,
        log_name="impulse-bitget",
        console_level="INFO",
        file_level="INFO",
        enable_console=False,
        enable_file=True,
        retention_days=60,
        config_dir=config_dir,
    )
    envelope_log = setup_live_logging(
        logfile=None,
        log_name="envelope-bitget",
        console_level="INFO",
        file_level="INFO",
        enable_console=False,
        enable_file=True,
        retention_days=60,
        config_dir=config_dir,
    )

    assert impulse_log == config_dir / "logs" / "impulse-bitget.log"
    assert envelope_log == config_dir / "logs" / "envelope-bitget.log"


def test_uses_custom_path_when_provided(tmp_path: Path):
    custom_path = tmp_path / "custom_live.log"

    logfile = setup_live_logging(
        logfile=custom_path,
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
    )

    assert logfile == custom_path
    assert logfile.exists()


def test_creates_parent_directories_for_custom_path(tmp_path: Path):
    logfile = tmp_path / "deeply" / "nested" / "live.log"

    assert not logfile.parent.exists()

    setup_live_logging(
        logfile=logfile,
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
    )

    assert logfile.parent.exists()
    assert logfile.exists()


def test_uses_timed_rotating_handler(tmp_path: Path):
    setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
    )

    file_handler = next(
        h
        for h in logging.root.handlers
        if isinstance(h, logging.handlers.TimedRotatingFileHandler)
    )
    assert file_handler is not None


def test_sets_default_retention_to_60_days(tmp_path: Path):
    setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
    )

    file_handler = next(
        h
        for h in logging.root.handlers
        if isinstance(h, logging.handlers.TimedRotatingFileHandler)
    )
    assert file_handler.backupCount == 60


@pytest.mark.parametrize(
    "retention_days",
    [
        30,
        60,
        90,
    ],
)
def test_respects_retention_days(tmp_path: Path, retention_days):
    setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=retention_days,
    )

    file_handler = next(
        h
        for h in logging.root.handlers
        if isinstance(h, logging.handlers.TimedRotatingFileHandler)
    )
    assert file_handler.backupCount == retention_days


def test_configures_console_handler_by_default(tmp_path: Path):
    setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
    )

    console_handler = next(
        (
            h
            for h in logging.root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.TimedRotatingFileHandler)
        ),
        None,
    )
    assert console_handler is not None


def test_disables_console_when_requested(tmp_path: Path):
    setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=False,
        enable_file=True,
        retention_days=60,
    )

    assert len(logging.root.handlers) == 1
    assert isinstance(
        logging.root.handlers[0], logging.handlers.TimedRotatingFileHandler
    )


def test_logs_to_file(tmp_path: Path):
    logfile = setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
    )

    test_logger = logging.getLogger("test.module")
    test_logger.info("Test live message")

    for handler in logging.root.handlers:
        handler.flush()

    log_content = logfile.read_text()
    assert "Test live message" in log_content


@pytest.mark.parametrize(
    ("console_level", "expected_level"),
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
    ],
)
def test_respects_console_level(tmp_path: Path, console_level, expected_level):
    setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level=console_level,
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
    )

    console_handler = next(
        h
        for h in logging.root.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.handlers.TimedRotatingFileHandler)
    )
    assert console_handler.level == expected_level


def test_does_not_disable_existing_loggers(tmp_path: Path, capsys):
    test_logger = logging.getLogger("test.module")

    setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
    )

    test_logger.info("Test message")

    captured = capsys.readouterr()
    assert "Test message" in captured.out


def test_sets_ccxt_logger_to_warning(tmp_path: Path, capsys):
    setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level="DEBUG",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
    )

    ccxt_logger = logging.getLogger("ccxt")
    ccxt_logger.debug("Should not appear")
    ccxt_logger.warning("Should appear")

    captured = capsys.readouterr()
    assert "Should not appear" not in captured.out
    assert "Should appear" in captured.out


def test_accepts_empty_custom_handlers_dict(tmp_path: Path):
    setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
        custom_handlers={},
    )

    assert len(logging.root.handlers) == 2


def test_accepts_none_custom_handlers(tmp_path: Path):
    setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
        custom_handlers=None,
    )

    assert len(logging.root.handlers) == 2


def test_adds_custom_handler_to_root_handlers(tmp_path: Path):
    custom_handlers = {
        "custom": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        }
    }

    setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
        custom_handlers=custom_handlers,
    )

    assert len(logging.root.handlers) == 3


def test_adds_multiple_custom_handlers(tmp_path: Path):
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

    setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=True,
        retention_days=60,
        custom_handlers=custom_handlers,
    )

    assert len(logging.root.handlers) == 4


def test_custom_handlers_work_without_console(tmp_path: Path):
    custom_handlers = {
        "custom": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        }
    }

    setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=False,
        enable_file=True,
        retention_days=60,
        custom_handlers=custom_handlers,
    )

    assert len(logging.root.handlers) == 2


def test_custom_handlers_work_without_file(tmp_path: Path):
    custom_handlers = {
        "custom": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        }
    }

    setup_live_logging(
        logfile=tmp_path / "test.log",
        log_name="a-bot",
        console_level="INFO",
        file_level="INFO",
        enable_console=True,
        enable_file=False,
        retention_days=60,
        custom_handlers=custom_handlers,
    )

    assert len(logging.root.handlers) == 2
