import logging
from pathlib import Path

import pytest

from robottraderslab._core import setup_notebook_logging
from robottraderslab._core.logging import is_notebook


def test_creates_timestamped_file_in_logs_folder_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    logfile = setup_notebook_logging()

    assert logfile.exists()
    assert logfile.parent.name == "logs"
    assert logfile.name.startswith("notebook_")
    assert logfile.name.endswith(".log")


def test_uses_custom_path_when_provided(tmp_path: Path):
    custom_path = tmp_path / "custom_notebook.log"

    logfile = setup_notebook_logging(
        logfile=custom_path,
        console_level="INFO",
        file_level="INFO",
        enable_file=True,
    )

    assert logfile == custom_path
    assert logfile.exists()


def test_creates_parent_directories_for_custom_path(tmp_path: Path):
    logfile = tmp_path / "deeply" / "nested" / "notebook.log"

    assert not logfile.parent.exists()

    setup_notebook_logging(
        logfile=logfile,
        console_level="INFO",
        file_level="INFO",
        enable_file=True,
    )

    assert logfile.parent.exists()
    assert logfile.exists()


def test_always_configures_both_console_and_file_handlers(tmp_path: Path):
    setup_notebook_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_file=True,
    )

    assert len(logging.root.handlers) == 2
    handler_types = {type(h).__name__ for h in logging.root.handlers}
    assert "StreamHandler" in handler_types
    assert "FileHandler" in handler_types


def test_console_is_info_by_default(tmp_path: Path):
    setup_notebook_logging(logfile=tmp_path / "test.log")

    console_handler = next(
        h
        for h in logging.root.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    )
    assert console_handler.level == logging.INFO


def test_file_is_info_by_default(tmp_path: Path):
    setup_notebook_logging(logfile=tmp_path / "test.log")

    file_handler = next(
        h for h in logging.root.handlers if isinstance(h, logging.FileHandler)
    )
    assert file_handler.level == logging.INFO


def test_logs_to_console(tmp_path: Path, capsys):
    setup_notebook_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_file=True,
    )

    test_logger = logging.getLogger("test.module")
    test_logger.info("Test notebook message")

    captured = capsys.readouterr()
    assert "Test notebook message" in captured.out


def test_logs_to_file(tmp_path: Path):
    logfile = setup_notebook_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_file=True,
    )

    test_logger = logging.getLogger("test.module")
    test_logger.info("Test notebook message")

    for handler in logging.root.handlers:
        handler.flush()

    log_content = logfile.read_text()
    assert "Test notebook message" in log_content


@pytest.mark.parametrize(
    ("console_level", "expected_level"),
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
    ],
)
def test_respects_console_level(tmp_path: Path, console_level, expected_level):
    setup_notebook_logging(
        logfile=tmp_path / "test.log",
        console_level=console_level,
        file_level="INFO",
        enable_file=True,
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
def test_respects_file_level(tmp_path: Path, file_level, expected_level):
    setup_notebook_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level=file_level,
        enable_file=True,
    )

    file_handler = next(
        h for h in logging.root.handlers if isinstance(h, logging.FileHandler)
    )
    assert file_handler.level == expected_level


def test_does_not_disable_existing_loggers(tmp_path: Path, capsys):
    test_logger = logging.getLogger("test.module")

    setup_notebook_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_file=True,
    )

    test_logger.info("Test message")

    captured = capsys.readouterr()
    assert "Test message" in captured.out


def test_sets_ccxt_logger_to_warning(tmp_path: Path, capsys):
    setup_notebook_logging(
        logfile=tmp_path / "test.log",
        console_level="DEBUG",
        file_level="INFO",
        enable_file=True,
    )

    ccxt_logger = logging.getLogger("ccxt")
    ccxt_logger.debug("Should not appear")
    ccxt_logger.warning("Should appear")

    captured = capsys.readouterr()
    assert "Should not appear" not in captured.out
    assert "Should appear" in captured.out


def test_accepts_empty_custom_handlers_dict(tmp_path: Path):
    setup_notebook_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_file=True,
        custom_handlers={},
    )

    assert len(logging.root.handlers) == 2


def test_accepts_none_custom_handlers(tmp_path: Path):
    setup_notebook_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_file=True,
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

    setup_notebook_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_file=True,
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

    setup_notebook_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_file=True,
        custom_handlers=custom_handlers,
    )

    assert len(logging.root.handlers) == 4


def test_custom_handlers_work_without_file(tmp_path: Path):
    custom_handlers = {
        "custom": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        }
    }

    setup_notebook_logging(
        logfile=tmp_path / "test.log",
        console_level="INFO",
        file_level="INFO",
        enable_file=False,
        custom_handlers=custom_handlers,
    )

    assert len(logging.root.handlers) == 2


def test_is_notebook_returns_false_in_regular_python():
    assert is_notebook() is False


def test_is_notebook_returns_true_for_zmq_interactive_shell(monkeypatch):
    class MockZMQShell:
        pass

    MockZMQShell.__name__ = "ZMQInteractiveShell"

    def mock_get_ipython():
        return MockZMQShell()

    import builtins

    monkeypatch.setattr(builtins, "get_ipython", mock_get_ipython, raising=False)

    from robottraderslab._core.logging.notebook_logging import (
        is_notebook as fresh_is_notebook,
    )

    assert fresh_is_notebook() is True


def test_is_notebook_returns_false_for_terminal_interactive_shell(monkeypatch):
    class MockTerminalShell:
        pass

    MockTerminalShell.__name__ = "TerminalInteractiveShell"

    def mock_get_ipython():
        return MockTerminalShell()

    import builtins

    monkeypatch.setattr(builtins, "get_ipython", mock_get_ipython, raising=False)

    from robottraderslab._core.logging.notebook_logging import (
        is_notebook as fresh_is_notebook,
    )

    assert fresh_is_notebook() is False
