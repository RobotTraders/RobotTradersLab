from typing import Protocol


class RateLimiterProtocol(Protocol):
    """Paces requests against a configured budget."""

    async def wait(self, endpoint: str) -> None:
        """Sleep until the endpoint's next free request slot.

        Args:
            endpoint: Endpoint the caller is about to hit; endpoints without
                a configured budget pass through without waiting.
        """
        ...
