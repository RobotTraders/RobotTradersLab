import asyncio
import json
import logging
from collections.abc import Callable, Iterable
from typing import IO, Any

from .file_lock import acquire_lock, open_state_file, release_lock, state_file_path

logger = logging.getLogger(__name__)

NO_WAIT = 0.0
MIN_WAIT_SECONDS = 0.001
_WINDOW_SECONDS = 1.0


class SharedState:
    """A handler attached at the reporting level carries every record to a
    notifier of its own, so the first failure to reach a scope's state is
    reported once.
    """

    def __init__(self, scope: str) -> None:
        self._path = state_file_path(scope)
        self._reported_unreachable = False

    async def under_lock(self, change: Callable[[IO[str]], float]) -> float:
        """The state only paces requests, so one that cannot be reached or locked
        lets the request through.

        Args:
            change: Returns the delay it decided on for the open state file.

        Returns:
            Seconds to sleep before asking again; `NO_WAIT` when the
            request may go.
        """
        try:
            with open_state_file(self._path) as state_file:
                if not await acquire_lock(state_file):
                    self._report(f"another process holds the lock on {self._path}")
                    return NO_WAIT
                try:
                    return change(state_file)
                finally:
                    release_lock(state_file)
        except OSError as error:
            logger.debug("Shared state at %s is unreachable: %s", self._path, error)
            self._report(error)
            return NO_WAIT

    async def wait_until_claimed(self, change: Callable[[IO[str]], float]) -> None:
        while True:
            delay = await self.under_lock(change)
            if delay == NO_WAIT:
                return
            await asyncio.sleep(delay)

    def _report(self, cause: OSError | str) -> None:
        if self._reported_unreachable:
            return
        self._reported_unreachable = True
        logger.error(
            "The budget on %s is not shared between processes for now: %s",
            self._path,
            cause,
        )


def claim_in_window(
    stamps: Iterable[float], cap: int, now: float
) -> tuple[float, list[float]]:
    """The caller stores the returned stamps only when the claim was taken.

    Args:
        stamps: The claim times of one key.
        cap: Claims the second holds at most.
        now: The monotonic reading of this claim.

    Returns:
        Seconds until the cap has room, `NO_WAIT` when it has, and the
        stamps still inside the window, `now` among them when the claim
        was taken.
    """
    live = [stamp for stamp in stamps if now - _WINDOW_SECONDS < stamp <= now]
    if len(live) >= cap:
        return max(min(live) + _WINDOW_SECONDS - now, MIN_WAIT_SECONDS), live
    live.append(now)
    return NO_WAIT, live


def read_state[State](
    state_file: IO[str], parse: Callable[[Any], State], empty: State
) -> State:
    """The state is only pacing data, so a corrupt file must never block
    trading.

    Args:
        state_file: The open budget file.
        parse: Builds the state from the decoded JSON; an AttributeError,
            KeyError, TypeError or ValueError from it marks the file
            unreadable.
        empty: The state an unreadable file is taken to hold.
    """
    state_file.seek(0)
    content = state_file.read()
    if not content:
        return empty
    try:
        return parse(json.loads(content))
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        logger.warning(
            "Resetting unreadable budget state at %s: %s", state_file.name, error
        )
        return empty


def write_state(state_file: IO[str], payload: object) -> None:
    """The file is truncated before the dump, so a crash mid-write leaves
    content `read_state` resets.

    Args:
        state_file: The open budget file.
        payload: The JSON-serialisable state to store.
    """
    state_file.seek(0)
    state_file.truncate()
    json.dump(payload, state_file)
    state_file.flush()
