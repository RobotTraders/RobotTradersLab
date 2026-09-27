import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import IO, NamedTuple

from .file_lock import acquire_lock, release_lock, state_file_path

logger = logging.getLogger(__name__)

_NO_WAIT = 0.0
_MIN_WAIT_SECONDS = 0.001
_HELD_WINDOW = "held"


class _Bucket(NamedTuple):
    remaining: int
    window: str
    reset_at: float


class SharedVenueWindow:
    """Venue-announced allowances shared by every process on the machine.

    Meant for venues that publish no figure and announce, on every response,
    how much of a key's allowance is left and when it resets. The
    announcement is the only source of the budget.

    An allowance is spent in the same hold that finds it available, at the
    moment the request is about to leave, so a caller delayed between deciding
    to send and sending never spends a claim the venue has already stopped
    counting. A deadline outlives the boot that recorded it, and a monotonic
    reading means nothing outside its own boot, so deadlines are stamped on
    the wall clock; each one counts from the delay the venue states, never
    from the absolute instant it names, which the two clocks' disagreement
    makes unreliable.

    A response carries no record of the claims made since it left, so an
    announcement for the window already recorded lowers a count and never
    raises it, and a key at zero stands until its deadline passes: what the
    venue said last outranks an answer that was already in flight.

    The state coordinates processes and settles nothing about a request, so a
    machine that cannot supply it leaves the caller unpaced on this budget.
    The first budget that cannot be reached is reported once, since a handler
    attached at this level carries every warning to a notifier of its own.
    """

    def __init__(self, scope: str) -> None:
        """Initialise the window.

        Args:
            scope: Name of the shared state; windows created with the same
                scope draw on one another's claims.
        """
        self._state_path = state_file_path(scope)
        self._reported_unreachable = False

    async def hold(self, key: str, seconds: float) -> None:
        """Bar a key for a stated delay, whatever is recorded for it.

        A venue naming a delay has counted the request it refused, so the
        delay outranks the allowance it announced before it.
        """
        await self._under_lock(lambda state_file: _write_hold(state_file, key, seconds))

    async def record(
        self, key: str, *, remaining: int, window: str, reset_after: float
    ) -> None:
        """Take in what a response announced about a key's allowance.

        Args:
            remaining: Requests the venue says are left in the window.
            window: The venue's own name for the window, which tells this
                announcement apart from one made about another window.
            reset_after: Seconds between this response and the reset.
        """
        announced = _Bucket(remaining, window, time.time() + reset_after)
        await self._under_lock(
            lambda state_file: _write_announcement(state_file, key, announced)
        )

    async def wait(self, key: str) -> None:
        while True:
            delay = await self._under_lock(
                lambda state_file: _delay_or_claim(state_file, key)
            )
            if delay == _NO_WAIT:
                return
            await asyncio.sleep(delay)

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

    def _report(self, cause: object) -> None:
        if self._reported_unreachable:
            return
        self._reported_unreachable = True
        logger.warning(
            "The budget on %s is not shared between processes for now: %s",
            self._state_path,
            cause,
        )


def _write_hold(state_file: IO[str], key: str, seconds: float) -> float:
    buckets = _read_state(state_file)
    buckets[key] = _Bucket(0, _HELD_WINDOW, time.time() + seconds)
    _write_state(state_file, buckets)
    return _NO_WAIT


def _write_announcement(state_file: IO[str], key: str, announced: _Bucket) -> float:
    buckets = _read_state(state_file)
    buckets[key] = _reconciled(buckets.get(key), announced)
    _write_state(state_file, buckets)
    return _NO_WAIT


def _reconciled(recorded: _Bucket | None, announced: _Bucket) -> _Bucket:
    if recorded is None:
        return announced
    if recorded.remaining <= 0 and time.time() < recorded.reset_at:
        return recorded
    if recorded.window != announced.window:
        return announced
    return announced._replace(remaining=min(recorded.remaining, announced.remaining))


def _delay_or_claim(state_file: IO[str], key: str) -> float:
    buckets = _read_state(state_file)
    bucket = buckets.get(key)
    if bucket is None:
        return _NO_WAIT
    now = time.time()
    if now >= bucket.reset_at:
        del buckets[key]
        _write_state(state_file, buckets)
        return _NO_WAIT
    if bucket.remaining <= 0:
        return max(bucket.reset_at - now, _MIN_WAIT_SECONDS)
    buckets[key] = bucket._replace(remaining=bucket.remaining - 1)
    _write_state(state_file, buckets)
    return _NO_WAIT


def _read_state(state_file: IO[str]) -> dict[str, _Bucket]:
    """The state is only pacing data, so a corrupt file must never block
    trading.
    """
    state_file.seek(0)
    content = state_file.read()
    if not content:
        return {}
    try:
        return {
            key: _Bucket(
                int(bucket["remaining"]),
                str(bucket["window"]),
                float(bucket["reset_at"]),
            )
            for key, bucket in json.loads(content).items()
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        logger.warning("Resetting unreadable budget state at %s", state_file.name)
        return {}


def _write_state(state_file: IO[str], buckets: dict[str, _Bucket]) -> None:
    state_file.seek(0)
    state_file.truncate()
    json.dump({key: bucket._asdict() for key, bucket in buckets.items()}, state_file)
    state_file.flush()
