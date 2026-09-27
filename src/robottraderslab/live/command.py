import logging
from pathlib import Path
from typing import Any

from robottraderslab._core import EX_TEMPFAIL, hold_instance_lock
from robottraderslab.bootstrap import (
    BotConfig,
    LiveConfig,
    load_log_handlers,
    load_secrets,
    report_unavailable_notifiers,
)

from .logging import setup_live_logging
from .runners import require_live_config, run_livebot

logger = logging.getLogger(__name__)

type _CustomHandlers = dict[str, dict[str, Any]]


def main(
    config: Path,
    logfile_override: Path | None,
    console: bool,
    logfiles_retention_days: int | None,
) -> int:
    """Trade one cycle on the candles that have closed since the last run.

    One instance of a config trades at a time, since two would trade its
    accounts twice.

    Returns:
        The process exit code.

    Raises:
        StrategyCriticalError: If the config is unusable.
        ExchangeCriticalError: If the venue cannot be traded on.
    """
    bot_config = BotConfig.from_file(config)
    live_config = require_live_config(bot_config)

    handlers, unavailable = load_log_handlers(
        live_config.notifier, load_secrets(bot_config.secrets_file)
    )
    _start_logging(
        live_config,
        bot_config.config_dir,
        logfile_override,
        console,
        logfiles_retention_days,
        handlers,
        f"live_{config.stem}",
    )
    report_unavailable_notifiers(unavailable)
    with hold_instance_lock(config) as held:
        if not held:
            logger.warning(
                "%s is already trading in another process, exiting", config.name
            )
            return EX_TEMPFAIL
        run_livebot(bot_config)
    return 0


def _start_logging(
    live_config: LiveConfig,
    config_dir: Path | None,
    logfile_override: Path | None,
    console: bool,
    retention_days_override: int | None,
    custom_handlers: _CustomHandlers,
    log_name: str,
) -> None:
    logging_config = live_config.logging
    setup_live_logging(
        logfile=logfile_override or logging_config.logfile,
        console_level=logging_config.console_level,
        file_level=logging_config.file_level,
        enable_console=console,
        enable_file=logging_config.enable_file,
        retention_days=(
            retention_days_override
            if retention_days_override is not None
            else logging_config.retention_days
        ),
        custom_handlers=custom_handlers,
        config_dir=config_dir,
        log_name=log_name,
    )
