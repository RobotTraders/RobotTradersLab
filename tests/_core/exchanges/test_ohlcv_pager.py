import asyncio
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from robottraderslab.exceptions import ExchangeTransientError
from robottraderslab.exchanges import OhlcvData, fetch_pages, no_candles, page_bounds

PAGE_MS = 1000
PAGES_IN_FLIGHT_BOUND = 10
MORE_PAGES_THAN_THE_BOUND_END_MS = 24 * PAGE_MS
NO_LEAK_TIMEOUT_SECONDS = 5

type PageFetcher = Callable[[int, int], Awaitable[OhlcvData]]


def one_candle(page_start_ms: int) -> OhlcvData:
    return np.array([[float(page_start_ms), 1.0, 2.0, 0.5, 1.5, 10.0]])


class PagesInFlight:
    """Page fetcher that remembers how many of its pages ran at the same time."""

    def __init__(self) -> None:
        self.live = 0
        self.peak = 0

    async def fetch_page(self, page_start_ms: int, page_end_ms: int) -> OhlcvData:
        self.live += 1
        self.peak = max(self.peak, self.live)
        await asyncio.sleep(0)
        self.live -= 1
        return no_candles()


@pytest.fixture
def requested_pages() -> list[tuple[int, int]]:
    return []


@pytest.fixture
def record_page(requested_pages) -> PageFetcher:
    async def fetch_page(page_start_ms: int, page_end_ms: int) -> OhlcvData:
        requested_pages.append((page_start_ms, page_end_ms))
        return one_candle(page_start_ms)

    return fetch_page


@pytest.fixture
def pages_in_flight() -> PagesInFlight:
    return PagesInFlight()


class TestPageGrid:
    async def test_range_of_a_single_page(self, record_page, requested_pages):
        await fetch_pages(
            record_page,
            page_bounds(0, 0, PAGE_MS),
        )

        assert requested_pages == [(0, PAGE_MS)]

    async def test_range_spanning_several_pages(self, record_page, requested_pages):
        await fetch_pages(
            record_page,
            page_bounds(0, 2 * PAGE_MS, PAGE_MS),
        )

        assert sorted(requested_pages) == [
            (0, PAGE_MS),
            (PAGE_MS, 2 * PAGE_MS),
            (2 * PAGE_MS, 3 * PAGE_MS),
        ]

    async def test_end_between_two_page_starts(self, record_page, requested_pages):
        await fetch_pages(
            record_page,
            page_bounds(0, PAGE_MS + PAGE_MS // 2, PAGE_MS),
        )

        assert sorted(requested_pages) == [(0, PAGE_MS), (PAGE_MS, 2 * PAGE_MS)]

    async def test_end_before_start(self, record_page, requested_pages):
        rows = await fetch_pages(
            record_page,
            page_bounds(PAGE_MS, 0, PAGE_MS),
        )

        assert requested_pages == []
        np.testing.assert_array_equal(rows, no_candles())


class TestRowsReturned:
    async def test_rows_of_every_page(self, record_page):
        rows = await fetch_pages(
            record_page,
            page_bounds(0, 2 * PAGE_MS, PAGE_MS),
        )

        np.testing.assert_array_equal(np.sort(rows[:, 0]), [0.0, 1000.0, 2000.0])

    async def test_pages_that_came_back_empty(self):
        async def fetch_page(page_start_ms: int, page_end_ms: int) -> OhlcvData:
            return no_candles()

        rows = await fetch_pages(
            fetch_page,
            page_bounds(0, 2 * PAGE_MS, PAGE_MS),
        )

        np.testing.assert_array_equal(rows, no_candles())


class TestPageTaskBound:
    async def test_range_of_more_pages_than_the_bound(self, pages_in_flight):
        await fetch_pages(
            pages_in_flight.fetch_page,
            page_bounds(0, MORE_PAGES_THAN_THE_BOUND_END_MS, PAGE_MS),
        )

        assert pages_in_flight.peak == PAGES_IN_FLIGHT_BOUND

    async def test_pages_after_a_range_that_failed(self, pages_in_flight):
        async def fetch_page(page_start_ms: int, page_end_ms: int) -> OhlcvData:
            raise RuntimeError("dropped")

        with pytest.raises(RuntimeError):
            await fetch_pages(
                fetch_page,
                page_bounds(0, PAGE_MS, PAGE_MS),
            )

        async with asyncio.timeout(NO_LEAK_TIMEOUT_SECONDS):
            await fetch_pages(
                pages_in_flight.fetch_page,
                page_bounds(0, MORE_PAGES_THAN_THE_BOUND_END_MS, PAGE_MS),
            )

        assert pages_in_flight.peak == PAGES_IN_FLIGHT_BOUND

    async def test_pages_after_a_range_that_was_cancelled(self, pages_in_flight):
        never_answered = asyncio.Event()

        async def fetch_page(page_start_ms: int, page_end_ms: int) -> OhlcvData:
            await never_answered.wait()
            return no_candles()

        cancelled_range = asyncio.create_task(
            fetch_pages(
                fetch_page,
                page_bounds(0, MORE_PAGES_THAN_THE_BOUND_END_MS, PAGE_MS),
            )
        )
        await asyncio.sleep(0)
        cancelled_range.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_range

        async with asyncio.timeout(NO_LEAK_TIMEOUT_SECONDS):
            await fetch_pages(
                pages_in_flight.fetch_page,
                page_bounds(0, MORE_PAGES_THAN_THE_BOUND_END_MS, PAGE_MS),
            )

        assert pages_in_flight.peak == PAGES_IN_FLIGHT_BOUND


class TestFailures:
    @patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_a_page_rejected_through_four_attempts(self, mock_sleep):
        attempts = 0

        async def fetch_page(page_start_ms: int, page_end_ms: int) -> OhlcvData:
            nonlocal attempts
            attempts += 1
            if attempts < 5:
                raise ExchangeTransientError("429")
            return one_candle(page_start_ms)

        rows = await fetch_pages(
            fetch_page,
            page_bounds(0, 0, PAGE_MS),
        )

        assert attempts == 5
        assert len(rows) == 1

    @patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_a_page_rejected_on_every_attempt(self, mock_sleep):
        attempts = 0

        async def fetch_page(page_start_ms: int, page_end_ms: int) -> OhlcvData:
            nonlocal attempts
            attempts += 1
            raise ExchangeTransientError("429")

        with pytest.raises(ExchangeTransientError):
            await fetch_pages(
                fetch_page,
                page_bounds(0, 0, PAGE_MS),
            )

        assert attempts == 5

    async def test_a_page_failing_beside_a_page_still_running(self):
        started = 0
        cancelled = 0

        async def fetch_page(page_start_ms: int, page_end_ms: int) -> OhlcvData:
            nonlocal started, cancelled
            started += 1
            if page_start_ms == 0:
                raise RuntimeError("dropped")
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled += 1
                raise
            return no_candles()

        with pytest.raises(RuntimeError):
            await fetch_pages(
                fetch_page,
                page_bounds(0, PAGE_MS, PAGE_MS),
            )

        assert started == 2
        assert cancelled == 1
