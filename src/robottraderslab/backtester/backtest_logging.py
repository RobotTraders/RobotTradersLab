import logging
import logging.config
from datetime import datetime
from pathlib import Path

from robottraderslab._core import (
    LogLevel,
    get_formatter_config,
    lowest_handler_level,
)


def setup_backtest_logging(
    logfile: str | Path | None,
    log_name: str,
    console_level: LogLevel,
    file_level: LogLevel,
    enable_console: bool,
    enable_file: bool,
    custom_handlers: dict[str, dict] | None = None,
    config_dir: Path | None = None,
) -> Path:
    """Earlier runs stay available for comparison.

    Args:
        logfile: Defaults to a timestamped file under a logs/ directory when
            not given.
        log_name: Names the default log file, so bots sharing a directory
            keep their runs apart.
        custom_handlers: Additional handlers to configure (e.g. Discord, Telegram).
        config_dir: Workspace directory anchoring the default log, so a bot
            writes beside its own config wherever the run started.
    """
    if logfile is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = (config_dir or Path()) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        logfile_path = logs_dir / f"backtest_{log_name}_{timestamp}.log"
    else:
        logfile_path = Path(logfile)
        if logfile_path.parent != Path("."):
            logfile_path.parent.mkdir(parents=True, exist_ok=True)

    handlers_config = {}
    root_handlers = []

    if enable_file:
        handlers_config["file"] = {
            "class": "logging.FileHandler",
            "level": file_level,
            "formatter": "default",
            "filename": str(logfile_path),
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
        "loggers": {
            "ccxt": {
                "level": "WARNING",
            },
            "robottraderslab._core.actions.executor": {
                "level": "WARNING",
            },
        },
    }
    logging.config.dictConfig(config)

    return logfile_path
