import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from .exceptions import ExchangeRecoverableError, ExchangeTransientError

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_ATTEMPTS = 3
BASE_DELAY_SECONDS = 1.0


async def retry_on_transient(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY_SECONDS,
    retryable: type[ExchangeRecoverableError] = ExchangeTransientError,
) -> T:
    """Each delay is drawn uniformly under a ceiling that doubles per attempt,
    so callers rejected together land at different instants across the
    window. A first retry is routine flow control and leaves a breadcrumb;
    a repeated one warns, since the venue is straining by then.

    Args:
        max_attempts: Total attempts before giving up.
        base_delay: Ceiling of the first delay in seconds.

    Raises:
        ExchangeRecoverableError: The last error once every attempt is spent.
    """
    last_error: ExchangeRecoverableError | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await fn(*args)
        except retryable as e:
            last_error = e
            if attempt == max_attempts:
                break
            delay = random.uniform(0, base_delay * (2 ** (attempt - 1)))  # nosec B311
            logger.log(
                logging.DEBUG if attempt == 1 else logging.WARNING,
                "%s failed (attempt %d/%d), retrying in %.1fs: %s",
                _callable_name(fn),
                attempt,
                max_attempts,
                delay,
                e,
            )
            await asyncio.sleep(delay)

    raise last_error  # type: ignore[misc]


def _callable_name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__qualname__", repr(fn))
