import asyncio
import errno
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

_STATE_DIR_NAME = "robottraderslab-rate-limiter"
_STATE_FILE_MODE = "a+"
_LOCK_RETRY_SECONDS = 0.005
_LOCK_REGION_BYTES = 1
LOCK_WAIT_LIMIT_SECONDS = 1.0


def state_file_path(scope: str) -> Path:
    return Path(tempfile.gettempdir()) / _STATE_DIR_NAME / f"{scope}.json"


if sys.platform == "win32":
    import msvcrt

    def open_state_file(state_path: Path) -> IO[str]:
        """`%TEMP%` is the user's own, so nothing another user prepared stands
        in it.

        Raises:
            OSError: The storage cannot be reached.
        """
        state_path.parent.mkdir(parents=True, exist_ok=True)
        return open(state_path, _STATE_FILE_MODE)

    def try_lock(state_file: IO[str]) -> bool:
        """A caller told no leaves the state alone."""
        state_file.seek(0)
        try:
            msvcrt.locking(state_file.fileno(), msvcrt.LK_NBLCK, _LOCK_REGION_BYTES)
        except OSError as error:
            logger.debug("The lock on %s is not free: %s", state_file.name, error)
            return False
        return True

    def release_lock(state_file: IO[str]) -> None:
        state_file.seek(0)
        msvcrt.locking(state_file.fileno(), msvcrt.LK_UNLCK, _LOCK_REGION_BYTES)

else:
    import fcntl

    _STATE_FILE_PERMISSIONS = 0o600
    _verified_state_dirs: set[Path] = set()

    def open_state_file(state_path: Path) -> IO[str]:
        """The temporary directory is shared by every user of the machine, so a
        claim writes only into a directory the process's own user owns and
        never through a link.

        Raises:
            OSError: The directory belongs to another user, a link stands at
                the state path, or the storage cannot be reached.
        """
        directory = state_path.parent
        if directory not in _verified_state_dirs:
            _verify_directory(directory)
        return open(state_path, _STATE_FILE_MODE, opener=_open_without_following)

    def _open_without_following(path: str, flags: int) -> int:
        return os.open(path, flags | os.O_NOFOLLOW, _STATE_FILE_PERMISSIONS)

    def _verify_directory(directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        _refuse_another_users(directory)
        _verified_state_dirs.add(directory)

    def _refuse_another_users(directory: Path) -> None:
        owner = directory.lstat().st_uid
        user = os.geteuid()
        if owner != user:
            raise PermissionError(
                errno.EACCES,
                f"the state directory belongs to uid {owner} and the process "
                f"runs as uid {user}",
                str(directory),
            )

    def try_lock(state_file: IO[str]) -> bool:
        """A caller told no leaves the state alone."""
        try:
            fcntl.flock(state_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            logger.debug("The lock on %s is not free: %s", state_file.name, error)
            return False
        return True

    def release_lock(state_file: IO[str]) -> None:
        fcntl.flock(state_file.fileno(), fcntl.LOCK_UN)


async def acquire_lock(state_file: IO[str]) -> bool:
    """A process that dies drops its lock, and one that hangs while holding it
    would otherwise stall every process behind it, so the wait ends at a
    deadline measured on a clock no machine adjusts under it.

    Returns:
        True once the lock is held; False when `LOCK_WAIT_LIMIT_SECONDS`
        passes first. A caller told no has to leave the state alone.
    """
    deadline = time.monotonic() + LOCK_WAIT_LIMIT_SECONDS
    while not try_lock(state_file):
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_LOCK_RETRY_SECONDS)
    return True
