import asyncio

from robottraderslab._core.rate_limiting import (
    acquire_lock,
    release_lock,
    state_file_path,
)
from robottraderslab._core.rate_limiting.file_lock import LOCK_WAIT_LIMIT_SECONDS

SCOPE = "test-scope"
POLL_ALLOWANCE_SECONDS = 0.05
RELEASE_TIMEOUT_SECONDS = 2.0


async def test_a_second_holder_waits_until_the_first_releases():
    state_path = state_file_path(SCOPE)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    with open(state_path, "a+") as first, open(state_path, "a+") as second:
        await acquire_lock(first)
        waiting = asyncio.create_task(acquire_lock(second))
        await asyncio.sleep(POLL_ALLOWANCE_SECONDS)
        acquired_while_first_held = waiting.done()

        release_lock(first)
        await asyncio.wait_for(waiting, timeout=RELEASE_TIMEOUT_SECONDS)
        release_lock(second)

    assert not acquired_while_first_held


async def test_a_holder_that_never_releases_lets_the_waiter_go():
    state_path = state_file_path(SCOPE)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    with open(state_path, "a+") as held, open(state_path, "a+") as waiting:
        await acquire_lock(held)
        took_the_lock = await asyncio.wait_for(
            acquire_lock(waiting), timeout=LOCK_WAIT_LIMIT_SECONDS * 3
        )
        release_lock(held)

    assert not took_the_lock
