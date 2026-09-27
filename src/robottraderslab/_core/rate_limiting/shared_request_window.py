import asyncio
import json
import logging
import time
from typing import IO

from .file_lock import acquire_lock, release_lock, state_file_path

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 1.0
_NO_WAIT = 0.0
_MIN_WAIT_SECONDS = 0.001


class SharedRequestWindow:
    """Per-second request caps shared by every process on the machine.

    A cap is claimed in the same hold that finds room for it, at the moment
    the request is about to leave, so a caller delayed between deciding to
    send and sending never spends a claim the venue has already stopped
    counting. Waiting earns nothing back: a full window opens only when its
    oldest claim leaves the second being counted. Meant for venues that cap
    the requests arriving within each second.

    Claims are stamped on the machine's monotonic clock, which every process
    on the machine shares and nothing adjusts underneath a sleeping caller; a
    stamp from a previous boot reads as the future and is discarded.

    The state coordinates processes and settles nothing about a request, so a
    machine that cannot supply it leaves the caller unpaced on this budget.
    The first budget that cannot be reached is reported once, since a handler
    attached at this level carries every warning to a notifier of its own.
    """

    def __init__(self, scope: str, *, max_per_second: dict[str, int]) -> None:
        """Initialise the window.

        Args:
            scope: Name of the shared state; windows created with the same
                scope draw on one another's claims.
            max_per_second: Cap per key. A key absent from the mapping is
                never paced.
        """
        self._max_per_second = max_per_second
        self._state_path = state_file_path(scope)
        self._reported_unreachable = False

    async def wait(self, key: str) -> None:
        cap = self._max_per_second.get(key)
        if cap is None:
            return
        while True:
            delay = await self._claim(key, cap)
            if delay == _NO_WAIT:
                return
            await asyncio.sleep(delay)

    async def _claim(self, key: str, cap: int) -> float:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_path, "a+") as state_file:
                if not await acquire_lock(state_file):
                    self._report(
                        f"another process holds the lock on {self._state_path}"
                    )
                    return _NO_WAIT
                try:
                    return _delay_or_claim(state_file, key, cap)
                finally:
                    release_lock(state_file)
        except OSError as error:
            self._report(error)
            return _NO_WAIT

    def _report(self, cause: object) -> None:
        if self._reported_unreachable:
            return
        self._reported_unreachable = True
        logger.warning(
            "The budget on %s is not shared between processes for now: %s",
            self._state_path,
            cause,
        )


def _delay_or_claim(state_file: IO[str], key: str, cap: int) -> float:
    claims = _read_state(state_file)
    now = time.monotonic()
    live = [
        stamp for stamp in claims.get(key, []) if now - _WINDOW_SECONDS < stamp <= now
    ]
    if len(live) >= cap:
        return max(min(live) + _WINDOW_SECONDS - now, _MIN_WAIT_SECONDS)
    live.append(now)
    claims[key] = live
    _write_state(state_file, claims)
    return _NO_WAIT


def _read_state(state_file: IO[str]) -> dict[str, list[float]]:
    """The state is only pacing data, so a corrupt file must never block
    trading.
    """
    state_file.seek(0)
    content = state_file.read()
    if not content:
        return {}
    try:
        return {
            key: [float(stamp) for stamp in stamps]
            for key, stamps in json.loads(content).items()
        }
    except (AttributeError, TypeError, ValueError):
        logger.warning("Resetting unreadable budget state at %s", state_file.name)
        return {}


def _write_state(state_file: IO[str], claims: dict[str, list[float]]) -> None:
    state_file.seek(0)
    state_file.truncate()
    json.dump(claims, state_file)
    state_file.flush()
