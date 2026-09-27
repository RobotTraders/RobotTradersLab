import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence

import numpy as np

from ..async_bridge import drain_tasks
from ..candle_grid import candle_closes_after
from ..retry import retry_on_transient
from ..timeframes import TimeFrame
from .ohlcv_adapter_interface import OhlcvData, no_candles

type _OhlcvPageFetcher = Callable[[int, int], Awaitable[OhlcvData]]
type _PageBounds = tuple[int, int]

_MAX_PAGE_TASKS = 10
_PAGE_FETCH_ATTEMPTS = 5
_MILLISECONDS_PER_SECOND = 1000


def page_bounds(start_ms: int, end_ms: int, page_ms: int) -> list[_PageBounds]:
    """Cut a range into pages of one span.

    Args:
        start_ms: Start of the range, where the first page starts.
        end_ms: End of the range; the last page is the one starting at or
            before it.
        page_ms: Span a single page covers, as the source's page size allows.

    Returns:
        The start and end of every page, in order.
    """
    return [
        (page_start_ms, page_start_ms + page_ms)
        for page_start_ms in range(start_ms, end_ms + 1, page_ms)
    ]


async def fetch_pages(
    fetch_page: _OhlcvPageFetcher,
    pages: Sequence[_PageBounds],
) -> OhlcvData:
    """Fetch pages several at a time, each retried on its own.

    A page waits for a slot before it is given a task, so a range spanning tens
    of thousands of pages never holds a task per page. The rate requests leave
    at is the source's own concern.

    Args:
        fetch_page: Fetches one page, given its start and end in UTC
            milliseconds since the epoch.
        pages: Bounds of the pages to fetch, as `page_bounds` cuts them.

    Returns:
        Candles of every page, in the order the pages were given.
    """
    slots = asyncio.Semaphore(_MAX_PAGE_TASKS)
    tasks: list[asyncio.Task[OhlcvData]] = []
    try:
        for page_start_ms, page_end_ms in pages:
            await slots.acquire()
            page_task = asyncio.create_task(
                retry_on_transient(
                    fetch_page,
                    page_start_ms,
                    page_end_ms,
                    max_attempts=_PAGE_FETCH_ATTEMPTS,
                )
            )
            page_task.add_done_callback(lambda _page_task: slots.release())
            tasks.append(page_task)
        fetched = await asyncio.gather(*tasks)
    except BaseException:
        await drain_tasks(tasks)
        raise

    if not fetched:
        return no_candles()
    return np.concatenate(fetched)


def without_the_open_candle(rows: OhlcvData, timeframe: TimeFrame) -> OhlcvData:
    """Drop the last candle when its period has not elapsed yet.

    No indicator may see a candle whose values can still change, and only the
    newest candle can be that one. When it closes depends on the length of its
    own period, which for a month the calendar decides.

    Args:
        rows: Candles as returned by the source, oldest first.
        timeframe: Candle timeframe.

    Returns:
        The candles whose period has elapsed.
    """
    if rows.size == 0:
        return rows

    closes_at_ms = candle_closes_after(timeframe, int(rows[-1, 0]))
    if closes_at_ms > time.time() * _MILLISECONDS_PER_SECOND:
        return rows[:-1]
    return rows
