import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from robottraderslab._core import EX_TEMPFAIL, tool_file

logger = logging.getLogger(__name__)

_MEMORY_SUFFIX = ".launches.json"
_WRITING_SUFFIX = ".writing"
_PAUSE_MINUTES = (1, 2, 4, 5)


@dataclass(frozen=True)
class BotLaunch:
    """The launch a bot last failed on, and when the scheduler tries again."""

    exit_code: int
    failures: int
    next_attempt: datetime
    config_mtime_ns: int


class LaunchMemory:
    """What a registry's bots did last time, in a file clearing the logs leaves alone.

    Every scheduler run is a fresh process, so reporting a death once and
    pausing the bot that died both depend on state outliving the run.
    """

    def __init__(self, memory_file: Path, launches: dict[str, BotLaunch]):
        self._memory_file = memory_file
        self._launches = launches

    @classmethod
    def of_registry(cls, registry_file: Path) -> "LaunchMemory":
        """A memory that cannot be read counts as absent, so a corrupted file
        costs a pause and a repeated report, never a bot that stops running.
        """
        memory_file = tool_file(registry_file, _MEMORY_SUFFIX)
        return cls(memory_file, _stored_launches(memory_file))

    def forget_deleted_bots(self) -> bool:
        """A bot whose config is gone is never launched again, so the pause and
        the failure count standing against it are facts about nothing.
        """
        deleted = [key for key in self._launches if not Path(key).exists()]
        for key in deleted:
            del self._launches[key]
        return bool(deleted)

    def is_paused(self, bot_config: Path, now: datetime) -> bool:
        """A config saved since the pause began lifts it at once, since that edit
        is what a broken configuration is paused for.
        """
        launch = self._launches.get(_key(bot_config))
        if launch is None:
            return False
        if now >= launch.next_attempt:
            return False
        return bot_config.stat().st_mtime_ns == launch.config_mtime_ns

    def record_exit(
        self, bot_config: Path, exit_code: int, now: datetime
    ) -> BotLaunch | None:
        """Remember a bot's failed launch.

        Returns:
            The launch to report, or None when the bot exited the same way
            last time, so one failure is reported once however many runs it
            spans.
        """
        key = _key(bot_config)
        previous = self._launches.get(key)
        if previous is not None and previous.exit_code == exit_code:
            self._launches[key] = _launch(
                bot_config, exit_code, now, previous.failures + 1
            )
            return None
        self._launches[key] = _launch(bot_config, exit_code, now, failures=1)
        return self._launches[key]

    def record_success(self, bot_config: Path) -> bool:
        """A bot that was already running recovered from nothing, so a run of
        healthy bots has nothing to report.
        """
        return self._launches.pop(_key(bot_config), None) is not None

    def save(self) -> None:
        """A run interrupted while writing leaves the memory of the run before
        it readable.
        """
        self._memory_file.parent.mkdir(parents=True, exist_ok=True)
        writing = self._memory_file.with_name(self._memory_file.name + _WRITING_SUFFIX)
        writing.write_text(json.dumps(_written_launches(self._launches), indent=2))
        writing.replace(self._memory_file)


def _stored_launches(memory_file: Path) -> dict[str, BotLaunch]:
    if not memory_file.exists():
        return {}
    try:
        stored = json.loads(memory_file.read_text())
        if not isinstance(stored, dict):
            raise ValueError("it holds no table of bots")
        return {key: _stored_launch(entry) for key, entry in stored.items()}
    except (OSError, ValueError, TypeError, KeyError) as e:
        logger.warning("Starting with no memory of `%s`: %s", memory_file, e)
        return {}


def _stored_launch(entry: Any) -> BotLaunch:
    return BotLaunch(
        exit_code=int(entry["exit_code"]),
        failures=int(entry["failures"]),
        next_attempt=datetime.fromisoformat(entry["next_attempt"]),
        config_mtime_ns=int(entry["config_mtime_ns"]),
    )


def _key(bot_config: Path) -> str:
    return str(bot_config.resolve())


def _launch(
    bot_config: Path, exit_code: int, now: datetime, failures: int
) -> BotLaunch:
    return BotLaunch(
        exit_code=exit_code,
        failures=failures,
        next_attempt=_next_attempt(now, exit_code, failures),
        config_mtime_ns=bot_config.stat().st_mtime_ns,
    )


def _next_attempt(now: datetime, exit_code: int, failures: int) -> datetime:
    """A pause is measured from the minute the launch failed in, so that it ends
    on one of the scheduler's own minutes.
    """
    return now.replace(second=0, microsecond=0) + _pause(exit_code, failures)


def _pause(exit_code: int, failures: int) -> timedelta:
    if exit_code == EX_TEMPFAIL:
        return timedelta()
    return timedelta(minutes=_PAUSE_MINUTES[min(failures, len(_PAUSE_MINUTES)) - 1])


def _written_launches(launches: dict[str, BotLaunch]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "exit_code": launch.exit_code,
            "failures": launch.failures,
            "next_attempt": launch.next_attempt.isoformat(),
            "config_mtime_ns": launch.config_mtime_ns,
        }
        for key, launch in launches.items()
    }
