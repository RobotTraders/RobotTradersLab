import logging
import logging.config
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any

from robottraderslab._core import (
    LogLevel,
    get_formatter_config,
    lowest_handler_level,
)

RUN_LOG_FILENAME = "backtest.log"
_QUIETED_LOGGERS = ("ccxt", "robottraderslab._core.actions.executor")


def setup_backtest_logging(
    logfile: Path | None,
    console_level: LogLevel,
    file_level: LogLevel,
    enable_console: bool,
    custom_handlers: dict[str, dict[str, Any]] | None = None,
) -> None:
    """The root logger's handlers are replaced by the ones this run configures."""
    handlers_config: dict[str, dict[str, Any]] = {}
    root_handlers: list[str] = []

    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers_config["file"] = {
            "class": "logging.FileHandler",
            "level": file_level,
            "formatter": "default",
            "filename": str(logfile),
            "encoding": "utf-8",
        }
        root_handlers.append("file")

    if enable_console:
        handlers_config["console"] = {
            "class": "logging.StreamHandler",
            "level": console_level,
            "formatter": "default",
            "stream": "ext://sys.stdout",
        }
        root_handlers.append("console")

    if custom_handlers:
        handlers_config.update(custom_handlers)
        for handler_name in custom_handlers:
            if handler_name not in root_handlers:
                root_handlers.append(handler_name)

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": get_formatter_config(),
        "handlers": handlers_config,
        "root": {
            "level": lowest_handler_level(handlers_config),
            "handlers": root_handlers,
        },
        "loggers": {name: {"level": logging.WARNING} for name in _QUIETED_LOGGERS},
    }
    logging.config.dictConfig(config)


@contextmanager
def logger_levels(
    levels: Mapping[logging.Logger | logging.Handler, int],
) -> Iterator[None]:
    """Every logger and handler is back at the level it had once the block
    ends, whatever the block raises.

    Args:
        levels: The level each logger or handler holds for the block.
    """
    previous = {target: target.level for target in levels}
    for target, level in levels.items():
        target.setLevel(level)
    try:
        yield
    finally:
        for target, level in previous.items():
            target.setLevel(level)


@contextmanager
def recorded_log(level: LogLevel) -> Iterator[StringIO]:
    """A run started from Python is saved, or not, only once it has finished,
    so its log is held in memory until its folder is written.

    A caller that configured no logging keeps the warnings Python prints to
    stderr: `logging.lastResort` stays on the root beside this run's handler.
    The root's own handlers keep showing what the root let through before,
    so lowering the root for this run's handler adds no line to them.
    """
    handler = logging.StreamHandler(StringIO())
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(get_formatter_config()["default"]["format"]))
    root = logging.getLogger()
    added: list[logging.Handler] = [handler]
    if not root.handlers and logging.lastResort is not None:
        added.append(logging.lastResort)
    levels: dict[logging.Logger | logging.Handler, int] = {
        root: min(root.level, handler.level),
        **{
            caller_handler: max(caller_handler.level, root.level)
            for caller_handler in root.handlers
        },
        **{logging.getLogger(name): logging.WARNING for name in _QUIETED_LOGGERS},
    }
    with logger_levels(levels):
        for added_handler in added:
            root.addHandler(added_handler)
        try:
            yield handler.stream
        finally:
            for added_handler in added:
                root.removeHandler(added_handler)
