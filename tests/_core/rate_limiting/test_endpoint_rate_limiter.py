import asyncio

import pytest

from robottraderslab.exchanges import RateLimiter

ENDPOINT_LOW_RATE = "endpoint-1"
ENDPOINT_HIGH_RATE = "endpoint-2"
LOW_RATE_INTERVAL = 1.0
HIGH_RATE_INTERVAL = 0.1


@pytest.fixture
def rate_limiter() -> RateLimiter:
    return RateLimiter(
        {
            ENDPOINT_LOW_RATE: LOW_RATE_INTERVAL,
            ENDPOINT_HIGH_RATE: HIGH_RATE_INTERVAL,
        }
    )


async def test_no_wait_before_first_call(rate_limiter, virtual_sleeps):
    await rate_limiter.wait(ENDPOINT_LOW_RATE)

    virtual_sleeps.assert_not_called()


async def test_no_wait_if_not_configured(rate_limiter, virtual_sleeps):
    await rate_limiter.wait("/")
    await rate_limiter.wait("/")

    virtual_sleeps.assert_not_called()


async def test_no_wait_for_two_different_endpoints(rate_limiter, virtual_sleeps):
    await rate_limiter.wait(ENDPOINT_LOW_RATE)
    await rate_limiter.wait(ENDPOINT_HIGH_RATE)

    virtual_sleeps.assert_not_called()


async def test_wait_before_second_call(rate_limiter, virtual_sleeps):
    await rate_limiter.wait(ENDPOINT_LOW_RATE)

    await rate_limiter.wait(ENDPOINT_LOW_RATE)

    virtual_sleeps.assert_called_once_with(pytest.approx(LOW_RATE_INTERVAL))


async def test_wait_for_the_remaining_interval(rate_limiter, clock, virtual_sleeps):
    await rate_limiter.wait(ENDPOINT_LOW_RATE)
    clock.now += 0.2

    await rate_limiter.wait(ENDPOINT_LOW_RATE)

    virtual_sleeps.assert_called_once_with(pytest.approx(0.8))


async def test_no_wait_once_the_interval_has_passed(
    rate_limiter, clock, virtual_sleeps
):
    await rate_limiter.wait(ENDPOINT_HIGH_RATE)
    clock.now += 0.2

    await rate_limiter.wait(ENDPOINT_HIGH_RATE)

    virtual_sleeps.assert_not_called()


async def test_a_cancelled_wait_leaves_no_slot_behind(rate_limiter, virtual_sleeps):
    await rate_limiter.wait(ENDPOINT_LOW_RATE)
    cancelled_wait = asyncio.create_task(rate_limiter.wait(ENDPOINT_LOW_RATE))
    await asyncio.sleep(0)
    cancelled_wait.cancel()

    await rate_limiter.wait(ENDPOINT_LOW_RATE)

    delays = [sleep_call.args[0] for sleep_call in virtual_sleeps.call_args_list]
    assert delays == [pytest.approx(0), pytest.approx(LOW_RATE_INTERVAL)]


async def test_concurrent_calls_leave_one_interval_apart(
    rate_limiter, clock, virtual_sleeps
):
    start = clock.now

    await asyncio.gather(
        rate_limiter.wait(ENDPOINT_HIGH_RATE),
        rate_limiter.wait(ENDPOINT_HIGH_RATE),
        rate_limiter.wait(ENDPOINT_HIGH_RATE),
    )

    assert clock.now - start == pytest.approx(2 * HIGH_RATE_INTERVAL)
