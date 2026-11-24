import time
from collections.abc import Callable
from typing import IO, Any, NamedTuple

from .shared_state import (
    MIN_WAIT_SECONDS,
    NO_WAIT,
    SharedState,
    read_state,
    write_state,
)

_HELD_WINDOW = "held"


class _Bucket(NamedTuple):
    remaining: int
    window: str
    reset_at: float


class SharedVenueWindow:
    """Meant for venues that publish no figure and announce, on every response,
    how much of a key's allowance is left and when it resets.

    An allowance is spent in the same hold that finds it available, at the
    moment the request is about to leave, so a caller delayed between deciding
    to send and sending never spends a claim the venue has already stopped
    counting. A deadline outlives the boot that recorded it, and a monotonic
    reading means nothing outside its own boot, so deadlines are stamped on
    the wall clock. The venue's clock may disagree with this machine's, so only
    the delay it states is trusted.

    Within one window the smaller remaining wins, and a key at zero stands until
    its deadline passes, since a response carries no record of the claims made
    since it left.

    The state coordinates processes and settles nothing about a request, so a
    machine that cannot supply it leaves the caller unpaced on this budget.
    The first budget that cannot be reached is reported once, since a handler
    attached at the reporting level carries every record to a notifier of its
    own.
    """

    def __init__(self, scope: str) -> None:
        self._state = SharedState(scope)

    async def hold(self, key: str, seconds: float) -> None:
        """A venue naming a delay has counted the request it refused."""
        held = _Bucket(0, _HELD_WINDOW, time.time() + seconds)
        await self._state.under_lock(
            lambda state_file: _set_bucket(state_file, key, lambda _: held)
        )

    async def record(
        self, key: str, *, remaining: int, window: str, reset_after: float
    ) -> None:
        """
        Args:
            key: What the allowance is counted against.
            remaining: Requests the venue says are left in the window.
            window: The venue's own name for the window.
            reset_after: Seconds between this response and the reset.
        """
        announced = _Bucket(remaining, window, time.time() + reset_after)
        await self._state.under_lock(
            lambda state_file: _set_bucket(
                state_file, key, lambda recorded: _reconciled(recorded, announced)
            )
        )

    async def wait(self, key: str) -> None:
        await self._state.wait_until_claimed(
            lambda state_file: _delay_or_claim(state_file, key)
        )


def _set_bucket(
    state_file: IO[str], key: str, replace: Callable[[_Bucket | None], _Bucket]
) -> float:
    buckets: dict[str, _Bucket] = read_state(state_file, _parse_state, {})
    buckets[key] = replace(buckets.get(key))
    _write_state(state_file, buckets)
    return NO_WAIT


def _reconciled(recorded: _Bucket | None, announced: _Bucket) -> _Bucket:
    if recorded is None:
        return announced
    if recorded.remaining <= 0 and time.time() < recorded.reset_at:
        return recorded
    if recorded.window != announced.window:
        return announced
    return announced._replace(remaining=min(recorded.remaining, announced.remaining))


def _delay_or_claim(state_file: IO[str], key: str) -> float:
    buckets: dict[str, _Bucket] = read_state(state_file, _parse_state, {})
    bucket = buckets.get(key)
    if bucket is None:
        return NO_WAIT
    now = time.time()
    if now >= bucket.reset_at:
        del buckets[key]
        _write_state(state_file, buckets)
        return NO_WAIT
    if bucket.remaining <= 0:
        return max(bucket.reset_at - now, MIN_WAIT_SECONDS)
    buckets[key] = bucket._replace(remaining=bucket.remaining - 1)
    _write_state(state_file, buckets)
    return NO_WAIT


def _parse_state(stored: Any) -> dict[str, _Bucket]:
    return {
        key: _Bucket(
            int(bucket["remaining"]),
            str(bucket["window"]),
            float(bucket["reset_at"]),
        )
        for key, bucket in stored.items()
    }


def _write_state(state_file: IO[str], buckets: dict[str, _Bucket]) -> None:
    write_state(state_file, {key: bucket._asdict() for key, bucket in buckets.items()})
