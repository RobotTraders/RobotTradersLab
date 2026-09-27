from .shared_request_window import SharedRequestWindow

_SHARED_BUDGET_KEY = "shared"


class SharedRateLimiter:
    """Per-second cap over a family of endpoints, shared by every process on
    the machine.

    Every endpoint the limiter covers draws from one window, so the cap holds
    whatever the number of processes and whatever the number of paths they
    call. Meant for exchange allowances enforced per IP, which all processes
    behind one address consume together.
    """

    def __init__(self, scope: str, *, max_per_second: int, path_segment: str) -> None:
        """Initialise the limiter.

        Args:
            scope: Name of the shared budget; limiters created with the same
                scope draw from one another's claims.
            max_per_second: Requests the covered endpoints may put on the wire
                within any second, together.
            path_segment: Endpoints whose path contains this segment draw from
                the budget. Any other path is left to its own limiter.
        """
        self._path_segment = path_segment
        self._window = SharedRequestWindow(
            scope, max_per_second={_SHARED_BUDGET_KEY: max_per_second}
        )

    async def wait(self, endpoint: str) -> None:
        if self._path_segment in endpoint:
            await self._window.wait(_SHARED_BUDGET_KEY)
