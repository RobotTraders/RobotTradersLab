import time

import pytest

from robottraderslab.exchanges import SharedRateLimiter

COVERED_ENDPOINT = "/covered/endpoint-1"
OTHER_COVERED_ENDPOINT = "/covered/endpoint-2"
UNCOVERED_ENDPOINT = "/uncovered/endpoint-1"
PATH_SEGMENT = "/covered/"
SCOPE = "test-scope"
MAX_PER_SECOND = 2


def _make_limiter(scope: str = SCOPE) -> SharedRateLimiter:
    return SharedRateLimiter(
        scope,
        max_per_second=MAX_PER_SECOND,
        path_segment=PATH_SEGMENT,
    )


class TestSharedRateLimiterOnAVirtualClock:
    async def test_calls_within_the_cap(self, virtual_sleeps):
        limiter = _make_limiter()

        await limiter.wait(COVERED_ENDPOINT)
        await limiter.wait(COVERED_ENDPOINT)

        virtual_sleeps.assert_not_called()

    async def test_uncovered_path(self, virtual_sleeps):
        limiter = _make_limiter()

        await limiter.wait(UNCOVERED_ENDPOINT)
        await limiter.wait(UNCOVERED_ENDPOINT)
        await limiter.wait(UNCOVERED_ENDPOINT)

        virtual_sleeps.assert_not_called()

    async def test_covered_paths_share_one_budget(self, virtual_sleeps):
        limiter = _make_limiter()
        await limiter.wait(COVERED_ENDPOINT)
        await limiter.wait(OTHER_COVERED_ENDPOINT)

        await limiter.wait(COVERED_ENDPOINT)

        virtual_sleeps.assert_called_once_with(pytest.approx(1.0))


class TestSharedRateLimiterOnTheRealClock:
    async def test_one_request_more_than_the_cap_allows(self):
        limiter = _make_limiter()

        start = time.monotonic()
        for _ in range(MAX_PER_SECOND + 1):
            await limiter.wait(COVERED_ENDPOINT)
        elapsed = time.monotonic() - start

        assert elapsed >= 1.0
