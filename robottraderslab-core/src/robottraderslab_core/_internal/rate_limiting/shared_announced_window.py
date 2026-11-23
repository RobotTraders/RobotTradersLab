import time
from typing import IO, Any, NamedTuple

from .shared_state import (
    NO_WAIT,
    SharedState,
    claim_in_window,
    read_state,
    write_state,
)


class _Announced(NamedTuple):
    per_second: int
    window: int


class _State(NamedTuple):
    claims: dict[str, list[float]]
    announced: dict[str, _Announced]


class SharedAnnouncedWindow:
    """Meant for venues that count each endpoint's arrivals per second and state
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
    The state reports its first failure to reach it, or to obtain its lock, once,
    since a handler attached at the reporting level carries every record to a
    notifier of its own.
    """

    def __init__(self, scope: str, *, default_per_second: int) -> None:
        self._default_per_second = default_per_second
        self._state = SharedState(scope)
        self._recorded_here: dict[str, _Announced] = {}

    async def record(self, key: str, *, per_second: int, window: int) -> None:
        """The answers of one venue window reach the machine in any order; a later
        window supersedes an earlier one, which is how a grant that falls is
        followed.

        Args:
            key: The endpoint the answer was counted against.
            per_second: Requests a second the answer shows the venue grants
                at least.
            window: The venue's own number for the second the answer was
                counted in.
        """
        announced = _Announced(per_second, window)
        recorded = self._recorded_here.get(key)
        if recorded and _prevailing(recorded, announced) == recorded:
            return
        self._recorded_here[key] = announced
        await self._state.under_lock(
            lambda state_file: _write_announcement(state_file, key, announced)
        )

    async def wait(self, key: str) -> None:
        """Returns at once when the machine cannot supply the shared state."""
        await self._state.wait_until_claimed(
            lambda state_file: _delay_or_claim(
                state_file, key, self._default_per_second
            )
        )


def _prevailing(recorded: _Announced, announced: _Announced) -> _Announced:
    if announced.window < recorded.window:
        return recorded
    if announced.window == recorded.window:
        return max(recorded, announced)
    return announced


def _write_announcement(state_file: IO[str], key: str, announced: _Announced) -> float:
    state = read_state(state_file, _parse_state, _State({}, {}))
    recorded = state.announced.get(key)
    state.announced[key] = (
        announced if recorded is None else _prevailing(recorded, announced)
    )
    _write_state(state_file, state)
    return NO_WAIT


def _delay_or_claim(state_file: IO[str], key: str, default_per_second: int) -> float:
    state = read_state(state_file, _parse_state, _State({}, {}))
    announced = state.announced.get(key)
    cap = announced.per_second if announced else default_per_second
    delay, live = claim_in_window(state.claims.get(key, []), cap, time.monotonic())
    if delay == NO_WAIT:
        state.claims[key] = live
        _write_state(state_file, state)
    return delay


def _parse_state(stored: Any) -> _State:
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


def _write_state(state_file: IO[str], state: _State) -> None:
    write_state(state_file, {"claims": state.claims, "announced": state.announced})
