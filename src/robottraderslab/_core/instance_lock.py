import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from .rate_limiting import release_lock, try_lock
from .tool_files import tool_file, tool_root

logger = logging.getLogger(__name__)

_LOCK_SUFFIX = ".lock"


@contextmanager
def hold_instance_lock(owner: Path) -> Iterator[bool]:
    """Hold the process lock of a TOML file for the block, one holder machine-wide.

    The lock is advisory and lives in the operating system, so a holder that
    crashes releases it with its file handle and never leaves a stale lock
    behind. A bot launched by hand and by the scheduler is named by two
    spellings of one path, and both have to meet the same lock. What the holder
    records is what lets a later sweep tell an owner that is gone from one still
    there.

    Yields:
        Whether the lock is held. A caller told no leaves the owner's work to
        the process holding it.
    """
    resolved = owner.resolve()
    lock_file = _lock_file(resolved)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "a+", encoding="utf-8") as handle:
        held = try_lock(handle)
        if held:
            _record_owner(handle, resolved)
        try:
            yield held
        finally:
            if held:
                release_lock(handle)


def drop_dead_locks(owners: Iterable[Path]) -> None:
    """A lock still held names a run still going, and a lock whose owner is gone
    can never be met again, so what leaves is a file no process will open.

    Args:
        owners: What the caller still knows of. Their tool directories are
            swept whole, so a lock the caller has forgotten leaves with them.
    """
    for root in {tool_root(owner) for owner in owners}:
        _sweep(root)


def _lock_file(owner: Path) -> Path:
    return tool_file(owner, _LOCK_SUFFIX)


def _record_owner(handle: IO[str], owner: Path) -> None:
    handle.seek(0)
    handle.truncate()
    handle.write(str(owner))
    handle.flush()


def _sweep(root: Path) -> None:
    for lock_file in root.rglob(f"*{_LOCK_SUFFIX}"):
        _drop_if_dead(lock_file)


def _drop_if_dead(lock_file: Path) -> None:
    """No holder can be writing the owner while the lock is held, which is the
    one state it is safe to read in.
    """
    try:
        with open(lock_file, "r+", encoding="utf-8") as handle:
            if not try_lock(handle):
                return
            handle.seek(0)
            owner = handle.read().strip()
            release_lock(handle)
        if owner and not Path(owner).exists():
            lock_file.unlink()
            logger.debug("Dropped the lock of `%s`, which is gone", owner)
    except OSError as e:
        logger.warning("Cannot sweep the lock file `%s`: %s", lock_file, e)
