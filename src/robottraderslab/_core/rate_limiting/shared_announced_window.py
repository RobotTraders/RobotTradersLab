import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import IO, NamedTuple

from .file_lock import acquire_lock, release_lock, state_file_path

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 1.0
_NO_WAIT = 0.0
_MIN_WAIT_SECONDS = 0.001


class _Announced(NamedTuple):
    per_second: int
    window: int


class _State(NamedTuple):
    claims: dict[str, list[float]]
    announced: dict[str, _Announced]


class SharedAnnouncedWindow:
    """Per-second request caps that follow what the venue announces, shared by
    every process on the machine.

    Meant for venues that count each endpoint's arrivals per second and state
    on every response how much of that second was left. Every process on the
    machine reads the announcements, so a process started later never opens at
    the default.

    A cap is claimed in the same hold that finds room for it, at the moment
    the request is about to leave, so a caller delayed between deciding to
    send and sending never spends a claim the venue has already stopped
    counting. Claims are stamped on the machine's monotonic clock, which
    every process on the machine shares; a stamp from a previous boot reads as
    the future and is discarded.

    The state coordinates processes and settles nothing about a request, so a
    machine that cannot supply it leaves the caller unpaced on this budget.
    A window reports its first failure to reach the state once, since a log
    handler can carry every warning to a notifier of its own.
    """

    def __init__(self, scope: str, *, default_per_second: int) -> None:
        """Initialise the window.

        Args:
            scope: Name of the shared state; windows created with the same
                scope draw on one another's claims and announcements.
            default_per_second: Cap of a key no announcement has reached.
        """
        self._default_per_second = default_per_second
        self._state_path = state_file_path(scope)
        self._reported_unreachable = False
        self._recorded_here: dict[str, _Announced] = {}

    async def record(self, key: str, *, per_second: int, window: int) -> None:
        """The answers of one venue window reach the machine in any order and
        each supports a cap the venue grants at least, so the largest of a
        window stands; a later window supersedes an earlier one, which is how
        a grant that falls is followed.

        Args:
            key: The endpoint the answer was counted against.
            per_second: Requests a second the answer shows the venue grants
                at least.
            window: The venue's own number for the second the answer was
                counted in.
        """
        announced = _Announced(per_second, window)
        recorded = self._recorded_here.get(key)
        if recorded and _supersedes(recorded, announced) is recorded:
            return
        self._recorded_here[key] = announced
        await self._under_lock(
            lambda state_file: _write_announcement(state_file, key, announced)
        )

    async def wait(self, key: str) -> None:
        """Returns at once when the machine cannot supply the shared state."""
        while True:
            delay = await self._under_lock(
                lambda state_file: _delay_or_claim(
                    state_file, key, self._default_per_second
                )
            )
            if delay == _NO_WAIT:
                return
            await asyncio.sleep(delay)

    def _report(self, cause: object) -> None:
        if self._reported_unreachable:
            return
        self._reported_unreachable = True
        logger.warning(
            "The budget on %s is not shared between processes for now: %s",
            self._state_path,
            cause,
        )

    async def _under_lock(self, change: Callable[[IO[str]], float]) -> float:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_path, "a+") as state_file:
                if not await acquire_lock(state_file):
                    self._report(
                        f"another process holds the lock on {self._state_path}"
                    )
                    return _NO_WAIT
                try:
                    return change(state_file)
                finally:
                    release_lock(state_file)
        except OSError as error:
            self._report(error)
            return _NO_WAIT


def _supersedes(recorded: _Announced, announced: _Announced) -> _Announced:
    if announced.window < recorded.window:
        return recorded
    if announced.window == recorded.window:
        return max(recorded, announced)
    return announced


def _write_announcement(state_file: IO[str], key: str, announced: _Announced) -> float:
    state = _read_state(state_file)
    recorded = state.announced.get(key)
    state.announced[key] = (
        announced if recorded is None else _supersedes(recorded, announced)
    )
    _write_state(state_file, state)
    return _NO_WAIT


def _delay_or_claim(state_file: IO[str], key: str, default_per_second: int) -> float:
    state = _read_state(state_file)
    announced = state.announced.get(key)
    cap = announced.per_second if announced else default_per_second
    now = time.monotonic()
    live = [
        stamp
        for stamp in state.claims.get(key, [])
        if now - _WINDOW_SECONDS < stamp <= now
    ]
    if len(live) >= cap:
        return max(min(live) + _WINDOW_SECONDS - now, _MIN_WAIT_SECONDS)
    live.append(now)
    state.claims[key] = live
    _write_state(state_file, state)
    return _NO_WAIT


def _read_state(state_file: IO[str]) -> _State:
    """The state is only pacing data, so a corrupt file must never block
    trading.
    """
    state_file.seek(0)
    content = state_file.read()
    if not content:
        return _State({}, {})
    try:
        stored = json.loads(content)
        return _State(
            {
                key: [float(stamp) for stamp in stamps]
                for key, stamps in stored["claims"].items()
            },
            {
                key: _Announced(int(per_second), int(window))
                for key, (per_second, window) in stored["announced"].items()
            },
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        logger.warning("Resetting unreadable budget state at %s", state_file.name)
        return _State({}, {})


def _write_state(state_file: IO[str], state: _State) -> None:
    state_file.seek(0)
    state_file.truncate()
    state_file.write(json.dumps({"claims": state.claims, "announced": state.announced}))
    state_file.flush()
