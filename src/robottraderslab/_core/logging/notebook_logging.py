import logging
import logging.config
from datetime import datetime
from pathlib import Path

from .formatters import get_formatter_config
from .handler_levels import lowest_handler_level
from .log_level import LogLevel


def is_notebook() -> bool:
    """Detect if running in a Jupyter notebook or JupyterLab environment."""
    try:
        shell: str = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False


def auto_configure_notebook_logging() -> None:
    """Set up console logging for interactive notebook sessions.

    Lets notebook users get log output without configuring logging
    themselves. No-op outside a notebook or when logging is already
    configured, so an existing setup (the CLI shim or the user's own) is
    never overridden.
    """
    if is_notebook() and not logging.getLogger().hasHandlers():
        setup_notebook_logging(enable_file=False)


def setup_notebook_logging(
    logfile: str | Path | None = None,
    console_level: LogLevel = LogLevel.INFO,
    file_level: LogLevel = LogLevel.INFO,
    enable_file: bool = True,
    custom_handlers: dict[str, dict] | None = None,
) -> Path:
    """Configure logging for notebook development.

    Creates both console and file output.

    Args:
        logfile: Path to log file. If None, creates logs/notebook_YYYY-MM-DD_HH-MM-SS.log
                 in a logs/ folder.
        custom_handlers: Additional handlers to configure (e.g. Discord, Telegram)

    Returns:
        Path to the created log file
    """
    if logfile is not None:
        logfile_path = Path(logfile)
        if logfile_path.parent != Path("."):
            logfile_path.parent.mkdir(parents=True, exist_ok=True)
    elif enable_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        logfile_path = logs_dir / f"notebook_{timestamp}.log"
    else:
        logfile_path = Path("notebook.log")

    handlers_config = {
        "console": {
            "class": "logging.StreamHandler",
            "level": console_level,
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    }
    root_handlers = ["console"]

    if enable_file:
        handlers_config["file"] = {
            "class": "logging.FileHandler",
            "level": file_level,
            "formatter": "default",
            "filename": str(logfile_path),
            "encoding": "utf-8",
        }
        root_handlers.append("file")

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
