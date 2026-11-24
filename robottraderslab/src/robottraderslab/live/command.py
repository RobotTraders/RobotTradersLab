import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from robottraderslab._internal import DryRunFailure, failure_of
from robottraderslab.bootstrap import (
    BotConfig,
    LiveConfig,
    bot_file,
    expected_bot_file,
    load_log_handlers,
    load_secrets,
    report_unavailable_notifiers,
)
from robottraderslab.exceptions import BotCriticalError, ExchangeCriticalError

from .logging import setup_live_logging
from .runners import dry_run_livebot, require_live_config, run_alone, run_livebot_at
from .working_copy import (
    EDIT_REFUSED_VARIABLE,
    EVERY_SYMBOL,
    NO_SYMBOL,
    TouchedSymbols,
    bot_of_working_copy,
    read_working_copy,
    touched_symbols,
    working_copy_error,
    working_copy_file,
    write_working_copy,
)

logger = logging.getLogger(__name__)

type _CustomHandlers = dict[str, dict[str, Any]]

_LIVE_LOG_NAME = "live"


def main(
    folder: Path,
    logfile_override: Path | None,
    console: bool,
    logfiles_retention_days: int | None,
) -> int:
    """Trade one cycle on the candles that have closed since the last run.

    One instance of a config trades at a time, since two would trade its
    accounts twice. The bot keeps a working copy of the last `bot.toml` that
    passed verification in a folder the engine keeps within its own folder.
    A `bot.toml` equal to the copy runs as it is. Any other is first run once
    with nothing sent, and becomes the copy, then trades, when that run met no
    failure of what the edit touches. Otherwise the cycle logs each failure and runs the
    working copy, the symbols the edit touches in exit-only mode. The
    environment variable `RTLAB_EDIT_REFUSED`, which holds a sentence, refuses
    an edit with no dry run: the cycle logs the sentence as the edit's failure,
    writes no copy and runs the working copy. It changes nothing for a
    `bot.toml` equal to the copy, for a bot with no copy or when it is empty.
    With no working copy, the file is verified and copied when it passes, and
    runs whether it passes or not.

    Returns:
        The process exit code, 0 for a cycle that ran the working copy.

    Raises:
        BotCriticalError: If the config is unusable and no working copy stands
            in for it, or if the working copy cannot run.
        ExchangeCriticalError: If the venue cannot be traded on.
        OSError: If the `bot.toml` or its log file cannot be opened.
    """
    config = expected_bot_file(folder)
    cycle = _Cycle(
        config=config,
        copy_file=working_copy_file(config),
        moment=datetime.now(UTC),
        logfile_override=logfile_override,
        console=console,
        retention_days_override=logfiles_retention_days,
    )
    edited = _load_edited(folder, cycle)
    working_copy = read_working_copy(cycle.copy_file)
    if working_copy is None:
        loaded = _loaded(edited)
        return run_alone(
            cycle.config,
            partial(_verify_first, cycle, loaded.bot_config, loaded.content),
        )
    if edited.content == working_copy:
        loaded = _loaded(edited)
        return run_alone(
            cycle.config, partial(_run, loaded.bot_config, cycle, NO_SYMBOL)
        )
    refusal = os.environ.get(EDIT_REFUSED_VARIABLE) or None
    return _run_edited(cycle, edited, working_copy, refusal)


@dataclass(frozen=True, slots=True)
class _Edited:
    """The `bot.toml` a cycle read, loaded with its log started."""

    content: bytes
    bot_config: BotConfig


@dataclass(frozen=True, slots=True)
class _Unloaded:
    """The `bot.toml` a cycle could not load or start the log of.

    Attributes:
        content: The file's bytes, None when it could not be read.
        error: What stopped the read, the load or the start of its log.
    """

    content: bytes | None
    error: BaseException


@dataclass(frozen=True, slots=True)
class _Cycle:
    """What every step of one cycle of `rtlab live` reads.

    Attributes:
        config: Where the bot's `bot.toml` sits.
        copy_file: Where the bot's working copy sits.
        moment: The moment the cycle started, which it books the
            candles of.
        logfile_override: The log file the command line names, over
            the configuration's.
        console: Whether the log goes to the console.
        retention_days_override: The days of logs the command line
            keeps, over the configuration's.
    """

    config: Path
    copy_file: Path
    moment: datetime
    logfile_override: Path | None
    console: bool
    retention_days_override: int | None


def _load_edited(folder: Path, cycle: _Cycle) -> _Edited | _Unloaded:
    """Read `bot.toml` once, load it and start the log it declares."""
    try:
        config = bot_file(folder)
        content = config.read_bytes()
    except (BotCriticalError, OSError) as error:
        logger.debug("`bot.toml` of `%s` cannot be read: %s", folder, error)
        return _Unloaded(None, error)
    try:
        bot_config = BotConfig.from_bytes(content, config)
        _start_logging(bot_config, cycle)
    except (Exception, BotCriticalError, ExchangeCriticalError) as error:
        logger.debug("`%s` cannot be loaded: %s", config, error)
        return _Unloaded(content, error)
    return _Edited(content, bot_config)


def _loaded(edited: _Edited | _Unloaded) -> _Edited:
    """The `bot.toml` the cycle loaded, raising what stopped its load.

    Raises:
        BotCriticalError: If the file is missing or declares no bot the engine
            can run.
        OSError: If the file or its log cannot be opened.
    """
    if isinstance(edited, _Unloaded):
        raise edited.error
    return edited


def _verify_first(cycle: _Cycle, bot_config: BotConfig, content: bytes) -> int:
    """Copy a file that passes on a bot with no working copy, then run it."""
    _dry_run_then_copy(cycle, bot_config, content, EVERY_SYMBOL)
    return _run(bot_config, cycle, NO_SYMBOL)


def _run_edited(
    cycle: _Cycle,
    edited: _Edited | _Unloaded,
    working_copy: bytes,
    edit_refusal: str | None,
) -> int:
    """Verify a `bot.toml` that differs from the working copy, then run one of them.

    An edit refusal is the sentence of the scheduler that refused the edit, and
    it is the edit's first failure. An edit with a refusal is not dry run, and
    neither is one that failed to load: the cycle runs the working copy.
    """
    touched = touched_symbols(working_copy, edited.content)
    failures = _failures_before_a_dry_run(edited, edit_refusal)
    if isinstance(edited, _Edited) and not failures:
        return run_alone(
            cycle.config,
            partial(
                _verify, cycle, edited.bot_config, edited.content, working_copy, touched
            ),
        )
    fallback = _start_working_copy(cycle, working_copy)
    return run_alone(
        cycle.config, partial(_fall_back, cycle, fallback, failures, touched)
    )


def _failures_before_a_dry_run(
    edited: _Edited | _Unloaded, edit_refusal: str | None
) -> list[DryRunFailure]:
    failures = []
    if edit_refusal is not None:
        failures.append(DryRunFailure(None, edit_refusal, recoverable=False))
    if isinstance(edited, _Unloaded):
        failures.append(failure_of(edited.error, None))
    return failures


def _verify(
    cycle: _Cycle,
    edited: BotConfig,
    content: bytes,
    working_copy: bytes,
    touched: TouchedSymbols,
) -> int:
    """Dry run the edited file, then trade it once copied, or run the working copy."""
    failures = _dry_run_then_copy(cycle, edited, content, touched)
    if failures is None:
        return _run(edited, cycle, NO_SYMBOL)
    fallback = _start_working_copy(cycle, working_copy)
    return _fall_back(cycle, fallback, failures, touched)


def _dry_run_then_copy(
    cycle: _Cycle, bot_config: BotConfig, content: bytes, touched: TouchedSymbols
) -> list[DryRunFailure] | None:
    """Dry run the file and make it the working copy when the edit's part passed.

    Returns:
        None when the file became the working copy, else the failures of what
        the edit touches, none when the write of the copy failed.
    """
    failures = [
        failure
        for failure in dry_run_livebot(bot_config, cycle.moment)
        if touched.covers(failure.symbol)
    ]
    if not failures and write_working_copy(cycle.copy_file, content):
        return None
    return failures


def _fall_back(
    cycle: _Cycle,
    fallback: BotConfig,
    failures: Sequence[DryRunFailure],
    touched: TouchedSymbols,
) -> int:
    """Log why the edited file is not applied, then run the working copy.

    The failures are those of what the edit touches.

    Raises:
        BotCriticalError: If the working copy cannot run, naming the copy.
    """
    level = _level_of(failures)
    _report_failures(cycle, failures, level)
    _report_exit_only(cycle, touched, level)
    try:
        return _run(fallback, cycle, touched)
    except BotCriticalError as error:
        raise working_copy_error(cycle.copy_file, error) from error


def _start_working_copy(cycle: _Cycle, working_copy: bytes) -> BotConfig:
    """Load the working copy and start the log it declares.

    The copy loads as if it were `bot.toml`, so it names the same bot.

    Raises:
        BotCriticalError: If the working copy declares no bot the engine can
            run or no `[live]` section, naming the copy.
        OSError: If the log file's folder cannot be created.
    """
    fallback = bot_of_working_copy(working_copy, cycle.config, cycle.copy_file)
    try:
        _start_logging(fallback, cycle)
    except BotCriticalError as error:
        raise working_copy_error(cycle.copy_file, error) from error
    return fallback


def _level_of(failures: Sequence[DryRunFailure]) -> int:
    """WARNING when every failure is a read the venue or the candles did not answer."""
    only_reads = bool(failures) and all(failure.recoverable for failure in failures)
    return logging.WARNING if only_reads else logging.ERROR


def _report_failures(
    cycle: _Cycle, failures: Sequence[DryRunFailure], level: int
) -> None:
    for failure in failures:
        where = f"{failure.symbol}: " if failure.symbol is not None else ""
        logger.log(
            level, "`%s` is not applied: %s%s", cycle.config, where, failure.message
        )


def _report_exit_only(cycle: _Cycle, touched: TouchedSymbols, level: int) -> None:
    """Name the symbols the working copy runs in exit-only mode."""
    logger.log(
        level,
        "Running the working copy of `%s` with %s in exit-only mode until it passes",
        cycle.config,
        _named(touched),
    )


def _named(touched: TouchedSymbols) -> str:
    if touched.every_symbol:
        return "every symbol"
    if not touched.symbols:
        return "no symbol"
    return ", ".join(sorted(str(symbol) for symbol in touched.symbols))


def _start_logging(bot_config: BotConfig, cycle: _Cycle) -> None:
    """Start the log the bot declares, reporting each notifier it cannot build.

    Raises:
        BotCriticalError: If the bot declares no `[live]` section.
        OSError: If the log file's folder cannot be created.
    """
    live_config = require_live_config(bot_config)
    handlers, unavailable = load_log_handlers(
        live_config.notifier, load_secrets(bot_config.secrets_file)
    )
    _setup_logging(live_config, bot_config.config_dir, cycle, handlers)
    report_unavailable_notifiers(unavailable)


def _setup_logging(
    live_config: LiveConfig,
    config_dir: Path | None,
    cycle: _Cycle,
    custom_handlers: _CustomHandlers,
) -> None:
    logging_config = live_config.logging
    setup_live_logging(
        logfile=cycle.logfile_override or logging_config.logfile,
        console_level=logging_config.console_level,
        file_level=logging_config.file_level,
        enable_console=cycle.console,
        enable_file=logging_config.enable_file,
        retention_days=(
            cycle.retention_days_override
            if cycle.retention_days_override is not None
            else logging_config.retention_days
        ),
        custom_handlers=custom_handlers,
        config_dir=config_dir,
        log_name=_LIVE_LOG_NAME,
    )


def _run(bot_config: BotConfig, cycle: _Cycle, exit_only: TouchedSymbols) -> int:
    run_livebot_at(bot_config, cycle.moment, exit_only)
    return 0
