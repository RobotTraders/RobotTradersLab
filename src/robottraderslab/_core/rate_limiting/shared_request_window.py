import time
from typing import IO, Any

from .shared_state import (
    NO_WAIT,
    SharedState,
    claim_in_window,
    read_state,
    write_state,
)


class SharedRequestWindow:
    """A cap is claimed in the same hold that finds room for it, at the moment
    the request is about to leave, so a caller delayed between deciding to
    send and sending never spends a claim the venue has already stopped
    counting. Meant for venues that cap the requests arriving within each
    second.

    Claims are stamped on the machine's monotonic clock, which every process
    on the machine shares and nothing adjusts underneath a sleeping caller; a
    stamp from a previous boot reads as the future and is discarded.

    The state coordinates processes and settles nothing about a request, so a
    machine that cannot supply it leaves the caller unpaced on this budget.
    The first budget that cannot be reached is reported once, since a handler
    attached at the reporting level carries every record to a notifier of its
    own.
    """

    def __init__(self, scope: str, *, max_per_second: dict[str, int]) -> None:
        self._max_per_second = max_per_second
        self._state = SharedState(scope)

    async def wait(self, key: str) -> None:
        cap = self._max_per_second.get(key)
        if cap is None:
            return
        await self._state.wait_until_claimed(
            lambda state_file: _delay_or_claim(state_file, key, cap)
        )


def _delay_or_claim(state_file: IO[str], key: str, cap: int) -> float:
    claims: dict[str, list[float]] = read_state(state_file, _parse_state, {})
    delay, live = claim_in_window(claims.get(key, []), cap, time.monotonic())
    if delay == NO_WAIT:
        claims[key] = live
        write_state(state_file, claims)
    return delay


def _parse_state(stored: Any) -> dict[str, list[float]]:
    return {key: [float(stamp) for stamp in stamps] for key, stamps in stored.items()}
