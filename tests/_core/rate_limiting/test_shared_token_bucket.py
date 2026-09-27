import asyncio
import logging

import pytest

from robottraderslab._core.rate_limiting import state_file_path
from robottraderslab.exchanges import SharedTokenBucket

SCOPE = "test-scope"


def _make_bucket(scope: str = SCOPE) -> SharedTokenBucket:
    return SharedTokenBucket(scope, capacity=2.0, refill_per_second=1.0)


async def test_no_wait_while_the_budget_covers_the_weight(virtual_sleeps):
    await _make_bucket().consume(2.0)

    virtual_sleeps.assert_not_called()


async def test_waits_for_the_missing_weight_to_refill(virtual_sleeps):
    bucket = _make_bucket()
    await bucket.consume(2.0)

    await bucket.consume(2.0)

    virtual_sleeps.assert_called_once_with(pytest.approx(2.0))


async def test_a_weight_above_capacity_waits_for_a_full_bucket(virtual_sleeps):
    bucket = _make_bucket()
    await bucket.consume(2.0)

    await bucket.consume(3.0)

    virtual_sleeps.assert_called_once_with(pytest.approx(2.0))


async def test_a_cancelled_wait_leaves_no_claim_behind(virtual_sleeps):
    bucket = _make_bucket()
    await bucket.consume(2.0)
    cancelled_wait = asyncio.create_task(bucket.consume(2.0))
    await asyncio.sleep(0)
    cancelled_wait.cancel()

    await bucket.consume(2.0)

    delays = [sleep_call.args[0] for sleep_call in virtual_sleeps.call_args_list]
    assert delays == [pytest.approx(0), pytest.approx(2.0)]


async def test_claims_are_shared_between_instances(virtual_sleeps):
    await _make_bucket().consume(2.0)

    await _make_bucket().consume(1.0)

    virtual_sleeps.assert_called_once_with(pytest.approx(1.0))


async def test_scopes_are_isolated(virtual_sleeps):
    await _make_bucket(scope="scope-a").consume(2.0)

    await _make_bucket(scope="scope-b").consume(2.0)

    virtual_sleeps.assert_not_called()


async def test_refill_is_capped_at_capacity(clock, virtual_sleeps):
    bucket = _make_bucket()
    await bucket.consume(2.0)
    clock.now += 100.0

    await bucket.consume(2.0)
    await bucket.consume(1.0)

    virtual_sleeps.assert_called_once_with(pytest.approx(1.0))


@pytest.mark.parametrize(
    "unreadable_state",
    ["{not json", '{"tokens": 1.0}', '{"tokens": null, "stamp": 1.0}'],
)
async def test_corrupt_state_is_reset_to_a_full_budget(
    unreadable_state, virtual_sleeps
):
    bucket = _make_bucket()
    await bucket.consume(2.0)
    state_file_path(SCOPE).write_text(unreadable_state)

    await bucket.consume(2.0)

    virtual_sleeps.assert_not_called()


async def test_an_unreachable_budget_never_blocks(caplog, virtual_sleeps):
    state_file_path(SCOPE).parent.mkdir(parents=True, exist_ok=True)
    state_file_path(SCOPE).mkdir()
    bucket = _make_bucket()

    with caplog.at_level(logging.WARNING):
        await bucket.consume(2.0)
        await bucket.consume(2.0)

    virtual_sleeps.assert_not_called()
    assert caplog.text.count("not shared between processes") == 1
