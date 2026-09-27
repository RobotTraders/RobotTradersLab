from collections.abc import Iterable

from .rate_limiter_protocol import RateLimiterProtocol


class ChainedRateLimiter:
    """Lets one client combine budgets with different scopes, such as
    machine-shared budgets for endpoints the exchange limits per IP and
    process-local budgets for endpoints it limits per account.
    """

    def __init__(self, limiters: Iterable[RateLimiterProtocol]) -> None:
        """Initialise the chain.

        Args:
            limiters: Held as a sequence from here on, so a one-shot iterable
                is safe to pass and every call still waits on all of them.
        """
        self._limiters = tuple(limiters)

    async def wait(self, endpoint: str) -> None:
        for limiter in self._limiters:
            await limiter.wait(endpoint)
