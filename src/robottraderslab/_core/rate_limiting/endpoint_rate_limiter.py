import asyncio
import time
from collections import defaultdict

_MIN_WAIT_SECONDS = 0.001


class RateLimiter:
    """Paces each endpoint to an interval of its own, over a budget this
    process is the only one drawing on.
    """

    def __init__(self, rate_limits: dict[str, float]) -> None:
        """Initialise the limiter.

        Args:
            rate_limits: Least seconds between calls, per endpoint. An
                endpoint absent from the mapping is never paced.
        """
        self._rate_limits = rate_limits
        self._next_slot_timestamp: dict[str, float] = defaultdict(float)

    async def wait(self, endpoint: str) -> None:
        """The slot is taken in the same step that finds it free, so a caller
        held up after a check never spends a slot it claimed earlier.
        """
        rate_limit = self._rate_limits.get(endpoint)
        if not rate_limit:
            return

        while True:
            now = time.time()
            slot = self._next_slot_timestamp[endpoint]
            if now >= slot:
                self._next_slot_timestamp[endpoint] = now + rate_limit
                return
            await asyncio.sleep(max(slot - now, _MIN_WAIT_SECONDS))
