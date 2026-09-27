import logging
from pathlib import Path
from typing import Any

from robottraderslab._core import LogLevel
from robottraderslab.bootstrap import (
    SecretsByName,
    load_log_handlers,
    load_secrets,
    refuse_credentials_in_notifiers,
    report_unavailable_notifiers,
)
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.live import setup_live_logging

from .scheduler import notifier_occasions, read_registry, run_scheduler

logger = logging.getLogger(__name__)

_SCHEDULER_LOG_NAME = "scheduler"
_LOG_RETENTION_DAYS = 60


def main(registry: Path) -> int:
    """Launch every registered bot whose candle closes this minute.

    Returns:
        How many bots exited on a failure.

    Raises:
        StrategyCriticalError: If the registry cannot be read.
    """
    entries: dict[str, Any] = {}
    notifier_config: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        entries = read_registry(registry)
        occasions = notifier_occasions(entries)
        refuse_credentials_in_notifiers("notifier", occasions)
        notifier_config = occasions
        _start_logging(
            registry,
            entries.get("logging", {}),
            notifier_config,
            _secrets(registry, entries),
        )
    except (StrategyCriticalError, RuntimeError, ValueError) as e:
        _start_logging(registry, {}, notifier_config, {})
        raise StrategyCriticalError(f"cannot start the scheduler: {e}") from e

    return run_scheduler(registry)


def _secrets(registry: Path, entries: dict[str, Any]) -> SecretsByName:
    secrets_file = entries.get("secrets_file")
    if secrets_file and not Path(secrets_file).is_absolute():
        secrets_file = registry.resolve().parent / secrets_file
    return load_secrets(secrets_file)


def _start_logging(
    registry: Path,
    logging_config: dict[str, Any],
    notifier_config: dict[str, dict[str, dict[str, Any]]],
    secrets_by_name: SecretsByName,
) -> None:
    handlers, unavailable = load_log_handlers(notifier_config, secrets_by_name)
    setup_live_logging(
        logfile=None,
        log_name=_SCHEDULER_LOG_NAME,
        config_dir=registry.parent,
        console_level=LogLevel(logging_config.get("console_level", LogLevel.INFO)),
        file_level=LogLevel(logging_config.get("file_level", LogLevel.INFO)),
        enable_console=logging_config.get("enable_console", False),
        enable_file=logging_config.get("enable_file", True),
        retention_days=logging_config.get("retention_days", _LOG_RETENTION_DAYS),
        custom_handlers=handlers,
    )
    report_unavailable_notifiers(unavailable)
