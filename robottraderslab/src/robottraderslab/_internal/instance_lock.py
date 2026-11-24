import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from robottraderslab_core.engine import release_lock, try_lock

from .tool_files import scheduler_file, tool_directory, tool_file, workspace_root

logger = logging.getLogger(__name__)

_LOCK_SUFFIX = ".lock"


@contextmanager
def hold_instance_lock(owner: Path) -> Iterator[bool]:
    """Hold the lock of the bot configured by `owner`, one holder machine-wide.

    The lock is advisory and lives in the operating system, so a holder that
    crashes releases it with its file handle and never leaves a stale lock
    behind. A bot launched by hand and by the scheduler is named by two
    spellings of one path, and both have to meet the same lock. What the holder
    records is what lets a later sweep tell an owner that is gone from one still
    there. The lock file is created in the `.rtlab` folder inside the bot's
    folder, which is created when missing.

    Yields:
        Whether the lock is held. A caller told no leaves the owner's work to
        the process holding it.
    """
    resolved = owner.resolve()
    with _hold(_lock_file(resolved), resolved) as held:
        yield held


@contextmanager
def hold_scheduler_lock(registry: Path) -> Iterator[bool]:
    """Hold the lock of the registry file for the block, one holder machine-wide.

    The lock is held as `hold_instance_lock` holds one, in the workspace's
    `.rtlab` folder: the nearest one at or above the registry's folder, or one
    made beside the registry when none exists.

    Yields:
        Whether the lock is held. A caller told no leaves the run to the
        process holding it.
    """
    resolved = registry.resolve()
    with _hold(_scheduler_lock_file(resolved), resolved) as held:
        yield held


def drop_dead_locks(
    registry: Path, bots: Iterable[Path], running_from_copy: Iterable[Path]
) -> None:
    """Delete the locks nothing opens again from the folders a scheduler sweeps.

    The folder holding the scheduler's own files for the registry, the lock
    folder beside each bot's file and the lock folder of each bot's
    workspace root are swept whole, and the empty folders under them are
    removed. The swept folders themselves stay. A lock still held names a run
    still going and is left. A lock is dead when the file it names is gone, or
    when it sits at neither path a run of that file opens: the lock folder
    beside the file, or, for a file that is no bot of the registry, the
    scheduler's place for it. A bot whose file is gone but which the scheduler
    still launches, from its working copy, is opened again, and its lock is kept.

    Args:
        registry: The scheduler's registry file.
        bots: The bot files the registry names, whether or not they still exist.
        running_from_copy: The bot files, among `bots`, that are gone while
            their bot runs from its working copy.
    """
    bot_files = _resolved(bots)
    bots_running_from_copy = _resolved(running_from_copy)
    roots = {_scheduler_lock_file(registry.resolve()).parent}
    for bot in bot_files:
        roots.update({_lock_file(bot).parent, tool_directory(workspace_root(bot))})
    for root in roots:
        for lock_file in root.rglob(f"*{_LOCK_SUFFIX}"):
            _drop_if_dead(lock_file, bot_files, bots_running_from_copy)
        for folder in sorted(root.rglob("*"), reverse=True):
            _drop_if_empty(folder)


@contextmanager
def _hold(lock_file: Path, owner: Path) -> Iterator[bool]:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "a+", encoding="utf-8") as handle:
        held = try_lock(handle)
        if held:
            _record_owner(handle, owner)
        try:
            yield held
        finally:
            if held:
                release_lock(handle)


def _lock_file(owner: Path) -> Path:
    return tool_file(owner, _LOCK_SUFFIX)


def _scheduler_lock_file(registry: Path) -> Path:
    return scheduler_file(registry, _LOCK_SUFFIX)


def _record_owner(handle: IO[str], owner: Path) -> None:
    handle.seek(0)
    handle.truncate()
    handle.write(str(owner))
    handle.flush()


def _resolved(paths: Iterable[Path]) -> set[Path]:
    return {path.resolve() for path in paths}


def _drop_if_dead(
    lock_file: Path, bot_files: set[Path], bots_running_from_copy: set[Path]
) -> None:
    """Drop a lock nothing opens again, reading its owner while the lock is held.

    No holder can be writing the owner while the lock is held, which is the one
    state it is safe to read in.
    """
    try:
        with open(lock_file, "r+", encoding="utf-8") as handle:
            if not try_lock(handle):
                return
            handle.seek(0)
            owner = handle.read().strip()
            release_lock(handle)
        if owner and not _is_opened_again(
            lock_file, Path(owner), bot_files, bots_running_from_copy
        ):
            lock_file.unlink()
            logger.debug("Dropped the lock of `%s`, which nothing opens again", owner)
    except OSError as e:
        logger.warning("Cannot sweep the lock file `%s`: %s", lock_file, e)


def _is_opened_again(
    lock_file: Path,
    owner: Path,
    bot_files: set[Path],
    bots_running_from_copy: set[Path],
) -> bool:
    if not owner.exists() and owner not in bots_running_from_copy:
        return False
    if lock_file == _lock_file(owner):
        return True
    return owner not in bot_files and lock_file == _scheduler_lock_file(owner)


def _drop_if_empty(folder: Path) -> None:
    try:
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    except OSError as e:
        logger.warning("Cannot remove the empty folder `%s`: %s", folder, e)
