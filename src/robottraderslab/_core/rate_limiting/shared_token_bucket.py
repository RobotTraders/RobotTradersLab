import time
from typing import IO, Any

from .shared_state import (
    MIN_WAIT_SECONDS,
    NO_WAIT,
    SharedState,
    read_state,
    write_state,
)


class SharedTokenBucket:
    """Independently started bots draw from one budget.

    A claim is made at the moment its
    request is about to leave, matching a venue whose own counter refills
    while callers hold back. Meant for exchange pools enforced per IP, which
    all processes behind one address consume together.

    The budget coordinates processes and settles nothing about a request, so a
    machine that cannot supply it leaves a caller pacing on whatever else it
    holds. The first budget that cannot be reached is reported once, since a
    handler attached at the reporting level carries every record to a notifier
    of its own.
    """

    def __init__(
        self, scope: str, *, capacity: float, refill_per_second: float
    ) -> None:
        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._state = SharedState(scope)

    async def consume(self, weight: float) -> None:
        """The weight is deducted in the same hold that finds it available,
        so a caller delayed after a check never spends a claim it made
        earlier.

        Args:
            weight: The cost of the request, in the bucket's units.
        """
        await self._state.wait_until_claimed(
            lambda state_file: self._delay_or_draw(state_file, weight)
        )

    def _delay_or_draw(self, state_file: IO[str], weight: float) -> float:
        tokens, stamp = read_state(
            state_file, _parse_state, (self._capacity, time.time())
        )
        now = time.time()
        tokens = min(self._capacity, tokens + (now - stamp) * self._refill_per_second)
        needed = min(weight, self._capacity)
        if tokens < needed:
            return max((needed - tokens) / self._refill_per_second, MIN_WAIT_SECONDS)
        write_state(state_file, {"tokens": tokens - weight, "stamp": now})
        return NO_WAIT


def _parse_state(stored: Any) -> tuple[float, float]:
    return float(stored["tokens"]), float(stored["stamp"])
