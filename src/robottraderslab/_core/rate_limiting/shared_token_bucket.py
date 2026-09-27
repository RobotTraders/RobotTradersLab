import asyncio
import json
import logging
import time
from typing import IO

from .file_lock import acquire_lock, release_lock, state_file_path

logger = logging.getLogger(__name__)

_NO_WAIT = 0.0
_MIN_WAIT_SECONDS = 0.001


class SharedTokenBucket:
    """Weight budget shared by every process on the machine.

    Claims live in a file guarded by an OS advisory lock, so independently
    started bots draw from one budget. A claim is made at the moment its
    request is about to leave, matching a venue whose own counter refills
    while callers hold back. Meant for exchange pools enforced per IP, which
    all processes behind one address consume together.

    The budget coordinates processes and settles nothing about a request, so a
    machine that cannot supply it leaves a caller pacing on whatever else it
    holds. The first budget that cannot be reached is reported once, since a
    handler attached at this level carries every warning to a notifier of its
    own.
    """

    def __init__(
        self, scope: str, *, capacity: float, refill_per_second: float
    ) -> None:
        """Initialise the bucket.

        Args:
            scope: Name of the shared budget; buckets created with the same
                scope share their reservations.
            capacity: Maximum weight the budget can hold.
            refill_per_second: Weight restored per second, up to capacity.
        """
        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._state_path = state_file_path(scope)
        self._reported_unreachable = False

    async def consume(self, weight: float) -> None:
        """The weight is deducted in the same hold that finds it available,
        so a caller delayed after a check never spends a claim it made
        earlier.
        """
        while True:
            delay = await self._claim(weight)
            if delay == _NO_WAIT:
                return
            await asyncio.sleep(delay)

    async def _claim(self, weight: float) -> float:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_path, "a+") as state_file:
                if not await acquire_lock(state_file):
                    self._report(
                        f"another process holds the lock on {self._state_path}"
                    )
                    return _NO_WAIT
                try:
                    return self._delay_or_draw(state_file, weight)
                finally:
                    release_lock(state_file)
        except OSError as error:
            self._report(error)
            return _NO_WAIT

    def _delay_or_draw(self, state_file: IO[str], weight: float) -> float:
        """A weight above capacity waits for a full bucket and draws into
        deficit.
        """
        tokens, stamp = _read_state(state_file, self._capacity)
        now = time.time()
        tokens = min(self._capacity, tokens + (now - stamp) * self._refill_per_second)
        needed = min(weight, self._capacity)
        if tokens < needed:
            return max((needed - tokens) / self._refill_per_second, _MIN_WAIT_SECONDS)
        _write_state(state_file, tokens - weight, now)
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


def _read_state(state_file: IO[str], capacity: float) -> tuple[float, float]:
    """The state is only pacing data, so a corrupt file must never block
    trading.
    """
    state_file.seek(0)
    content = state_file.read()
    if not content:
        return capacity, time.time()
    try:
        state = json.loads(content)
        return float(state["tokens"]), float(state["stamp"])
    except (ValueError, KeyError, TypeError):
        logger.warning("Resetting unreadable budget state at %s", state_file.name)
        return capacity, time.time()


def _write_state(state_file: IO[str], tokens: float, stamp: float) -> None:
    state_file.seek(0)
    state_file.truncate()
    json.dump({"tokens": tokens, "stamp": stamp}, state_file)
    state_file.flush()
