from unittest.mock import AsyncMock, Mock, call

from robottraderslab.exchanges import ChainedRateLimiter, RateLimiterProtocol

ENDPOINT = "/endpoint"


async def test_every_limiter_is_consulted_for_the_endpoint():
    first = AsyncMock(spec=RateLimiterProtocol)
    second = AsyncMock(spec=RateLimiterProtocol)

    await ChainedRateLimiter((first, second)).wait(ENDPOINT)

    first.wait.assert_awaited_once_with(ENDPOINT)
    second.wait.assert_awaited_once_with(ENDPOINT)


async def test_limiters_are_consulted_in_the_given_order():
    order_recorder = Mock()
    first = AsyncMock(spec=RateLimiterProtocol)
    second = AsyncMock(spec=RateLimiterProtocol)
    order_recorder.attach_mock(first, "first")
    order_recorder.attach_mock(second, "second")

    await ChainedRateLimiter((first, second)).wait(ENDPOINT)

    assert order_recorder.mock_calls == [
        call.first.wait(ENDPOINT),
        call.second.wait(ENDPOINT),
    ]
