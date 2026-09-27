import asyncio
import sys
import tempfile
import time
from pathlib import Path
from typing import IO

_STATE_DIR_NAME = "robottraderslab-rate-limiter"
_LOCK_RETRY_SECONDS = 0.005
_LOCK_REGION_BYTES = 1
LOCK_WAIT_LIMIT_SECONDS = 1.0


def state_file_path(scope: str) -> Path:
    """Return the machine-local state file backing a shared budget scope."""
    return Path(tempfile.gettempdir()) / _STATE_DIR_NAME / f"{scope}.json"


if sys.platform == "win32":
    import msvcrt

    def try_lock(state_file: IO[str]) -> bool:
        """Take the lock if it is free; a caller told no leaves the state alone."""
        state_file.seek(0)
        try:
            msvcrt.locking(state_file.fileno(), msvcrt.LK_NBLCK, _LOCK_REGION_BYTES)
        except OSError:
            return False
        return True

    def release_lock(state_file: IO[str]) -> None:
        """Release the advisory lock held on the file."""
        state_file.seek(0)
        msvcrt.locking(state_file.fileno(), msvcrt.LK_UNLCK, _LOCK_REGION_BYTES)

else:
    import fcntl

    def try_lock(state_file: IO[str]) -> bool:
        """Take the lock if it is free; a caller told no leaves the state alone."""
        try:
            fcntl.flock(state_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def release_lock(state_file: IO[str]) -> None:
        """Release the advisory lock held on the file."""
        fcntl.flock(state_file.fileno(), fcntl.LOCK_UN)


async def acquire_lock(state_file: IO[str]) -> bool:
    """Hold the event loop free while polling for the advisory lock.

    A process that dies drops its lock, and one that hangs while holding it
    would otherwise stall every process behind it, so the wait ends at a
    deadline measured on a clock no machine adjusts under it.

    Returns:
        Whether the lock is held. A caller told no has to leave the state
        alone.
    """
    deadline = time.monotonic() + LOCK_WAIT_LIMIT_SECONDS
    while not try_lock(state_file):
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_LOCK_RETRY_SECONDS)
    return True
