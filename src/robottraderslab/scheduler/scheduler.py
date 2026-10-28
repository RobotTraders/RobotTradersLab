import asyncio
import logging
import os
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robottraderslab._core import (
    EX_CONFIG,
    EX_TEMPFAIL,
    TimeFrame,
    candle_closes_at,
    drop_dead_locks,
    hold_instance_lock,
    run_async,
)
from robottraderslab.bootstrap import bot_file, bot_name, names_a_file, not_a_bot
from robottraderslab.exceptions import BotCriticalError

from .launch_memory import BotLaunch, LaunchMemory
from .shared_account import refuse_shared_account

logger = logging.getLogger(__name__)

_BOT_KEY = "bot"
_CONFIG_KEY = "config"
_EXCHANGES_KEY = "exchanges"
_NOTIFIER_KEY = "notifier"
_CADENCE_SECONDS = 60
_CLOCK_FORMAT = "%H:%M UTC"


@dataclass(frozen=True)
class _RegisteredBot:
    config: Path
    timeframes: list[TimeFrame]


def run_scheduler(registry_file: str | Path) -> int:
    """Launch every registered bot that is due in the current minute.

    Each due bot runs as its own process, so one failing bot does not stop
    the others. A bot registered with timeframes is due when one of them
    closes a candle in the current minute; a bot registered without
    timeframes is launched every run and gates itself. One run works a
    registry at a time, since two would launch the same bot and trade its
    account twice. A bot that died on its configuration waits out a pause
    before being launched again, so a fault nobody has fixed yet costs one
    report for as long as it stands. A bot deleted from the workspace takes
    the files the engine kept about it with it.

    Args:
        registry_file: TOML file with a `[[bot]]` array; each entry names
            its bot's folder as `config` and an optional `timeframes` list. An
            optional
            `[exchanges]` table and the scalars under `[notifier.<plugin>]`
            carry settings every launched bot receives.

    Returns:
        Number of launched bots that exited with an error. A bot that found
        another instance of itself trading is not one of them.

    Raises:
        BotCriticalError: If the registry cannot be read or parsed, names a
            file where a bot's folder belongs, or two
            of its bots trade one account under the same bot digest.
    """
    registry_path = Path(registry_file)
    registry = read_registry(registry_path)
    bots = _load_registered_bots(registry, registry_path)
    refuse_shared_account([bot.config for bot in bots])

    with hold_instance_lock(registry_path) as held:
        if not held:
            logger.warning(
                "Previous run still going, launching nothing from %s",
                registry_path.name,
            )
            return 0
        memory = LaunchMemory.of_registry(registry_path)
        _drop_what_deleted_bots_left(memory, registry_path, bots)
        if not bots:
            logger.warning("No bots registered in %s", registry_file)
            return 0
        return _launch_due_bots(registry, bots, memory)


def read_registry(registry_file: Path) -> dict[str, Any]:
    """Parse the registry file.

    A registry that cannot be read stops every bot at once, so the failure
    names the file it could not read.

    Raises:
        BotCriticalError: If the file is missing, unreadable or not
            valid TOML.
    """
    try:
        return tomllib.loads(registry_file.read_text())
    except FileNotFoundError as e:
        raise BotCriticalError(f"there is no registry file at `{registry_file}`") from e
    except OSError as e:
        raise BotCriticalError(
            f"cannot read the registry at `{registry_file}`: {e.strerror}"
        ) from e
    except tomllib.TOMLDecodeError as e:
        raise BotCriticalError(
            f"the registry at `{registry_file}` is not valid TOML: {e}"
        ) from e


def notifier_occasions(
    registry: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return the scheduler's own notifier subscriptions from the registry.

    A scalar under `[notifier.<plugin>]` is a setting of the plugin itself,
    so only the tables beneath a plugin subscribe an occasion.
    """
    return {
        plugin: {
            occasion: config
            for occasion, config in entries.items()
            if isinstance(config, dict)
        }
        for plugin, entries in registry.get(_NOTIFIER_KEY, {}).items()
    }


def _load_registered_bots(
    raw: dict[str, Any], registry_file: Path
) -> list[_RegisteredBot]:
    base_dir = registry_file.resolve().parent
    bots: list[_RegisteredBot] = []
    for position, entry in enumerate(raw.get(_BOT_KEY, []), start=1):
        folder_value = entry.get(_CONFIG_KEY)
        if not folder_value:
            logger.error(
                "Registry entry %s has no `%s`, skipping it", position, _CONFIG_KEY
            )
            continue
        folder = base_dir / folder_value
        refusal = not_a_bot(folder)
        if refusal is not None and names_a_file(folder):
            raise BotCriticalError(refusal)
        if refusal is not None:
            logger.error("%s, skipping it", refusal)
            continue
        bots.append(_RegisteredBot(bot_file(folder), entry.get("timeframes", [])))
    return bots


def _drop_what_deleted_bots_left(
    memory: LaunchMemory, registry_file: Path, bots: list[_RegisteredBot]
) -> None:
    """A bot deleted from the workspace leaves behind what the engine kept
    about it, and nothing ever comes back for it.
    """
    if memory.forget_deleted_bots():
        memory.save()
    drop_dead_locks([registry_file, *(bot.config for bot in bots)])


def _launch_due_bots(
    registry: dict[str, Any], bots: list[_RegisteredBot], memory: LaunchMemory
) -> int:
    now = datetime.now(UTC)
    due_configs = [bot.config for bot in bots if _is_due(bot, now)]
    if not due_configs:
        logger.debug("No registered bot is due, nothing to launch")
        return 0
    launchable = [
        config for config in due_configs if not _is_paused(memory, config, now)
    ]
    if not launchable:
        return 0
    os.environ.update(_shared_settings(registry))
    return run_async(_launch_bots(launchable, memory, now))


def _is_due(bot: _RegisteredBot, now: datetime) -> bool:
    if not bot.timeframes:
        return True
    return any(candle_closes_at(timeframe, now) for timeframe in bot.timeframes)


def _shared_settings(raw: dict[str, Any]) -> dict[str, str]:
    """Name the settings that reach every launched bot as environment variables.

    An exchange budget enforced per address covers every bot on the machine at
    once, so they stay within it only while they all pace to the same figure.
    A notifier plugin's setting is named once for the whole machine, as a
    scalar under `[notifier.<plugin>]`.
    """
    exchange_settings = {
        f"{exchange}_{setting}".upper(): str(value)
        for exchange, settings in raw.get(_EXCHANGES_KEY, {}).items()
        for setting, value in settings.items()
    }
    notifier_settings = {
        f"{plugin}_{setting}".upper(): str(value)
        for plugin, entries in raw.get(_NOTIFIER_KEY, {}).items()
        for setting, value in entries.items()
        if not isinstance(value, dict)
    }
    return {**exchange_settings, **notifier_settings}


async def _launch_bots(
    bot_configs: list[Path], memory: LaunchMemory, now: datetime
) -> int:
    outcomes = await asyncio.gather(
        *[_run_bot(config, memory, now) for config in bot_configs],
        return_exceptions=True,
    )
    failures = 0
    for bot_config, outcome in zip(bot_configs, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            logger.error("Cannot launch %s", bot_name(bot_config), exc_info=outcome)
            failures += 1
        elif _failed(outcome):
            failures += 1
    memory.save()
    return failures


async def _run_bot(bot_config: Path, memory: LaunchMemory, now: datetime) -> int:
    logger.info("Launching %s", bot_name(bot_config))
    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "robottraderslab",
        "live",
        str(bot_config.parent),
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    exit_code = await process.wait()
    _report_slow_cycle(bot_name(bot_config), time.monotonic() - started)
    _report_lifecycle(bot_config, exit_code, stderr, memory, now)
    return exit_code


def _failed(exit_code: int) -> bool:
    return exit_code not in (0, EX_TEMPFAIL)


def _is_paused(memory: LaunchMemory, bot_config: Path, now: datetime) -> bool:
    if not memory.is_paused(bot_config, now):
        return False
    logger.debug("%s is paused, not launching it", bot_name(bot_config))
    return True


def _report_slow_cycle(name: str, seconds: float) -> None:
    if seconds > _CADENCE_SECONDS:
        logger.warning(
            "%s took %.0f seconds, longer than the minute between runs",
            name,
            seconds,
        )


def _report_lifecycle(
    bot_config: Path, exit_code: int, stderr: bytes, memory: LaunchMemory, now: datetime
) -> None:
    """A bot with a log of its own has already named what killed it, so the only
    fact left for whoever launched it is what happens to the bot next.
    """
    if exit_code == 0:
        if memory.record_success(bot_config):
            logger.warning("%s is running again", bot_name(bot_config))
        return
    launch = memory.record_exit(bot_config, exit_code, now)
    if launch is not None:
        _report_failed_launch(bot_name(bot_config), launch, stderr)


def _report_failed_launch(name: str, launch: BotLaunch, stderr: bytes) -> None:
    if launch.exit_code == EX_TEMPFAIL:
        logger.warning("%s exited %s, retrying every run", name, launch.exit_code)
        return
    logger.error(
        "%s exited %s, paused, next attempt at %s%s",
        name,
        launch.exit_code,
        launch.next_attempt.strftime(_CLOCK_FORMAT),
        _carried_sentence(launch.exit_code, stderr),
    )


def _carried_sentence(exit_code: int, stderr: bytes) -> str:
    """A code that says the bot logged its own cause never carries it twice."""
    if exit_code == EX_CONFIG:
        return ""
    reason = stderr.decode(errors="replace").strip()
    return f": {reason}" if reason else ""
