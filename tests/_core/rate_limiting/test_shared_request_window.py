import asyncio
import json
import logging

import pytest

from robottraderslab._core.rate_limiting import (
    acquire_lock,
    release_lock,
    state_file_path,
)
from robottraderslab.exchanges import SharedRequestWindow

SCOPE = "test-scope"
PACED_KEY = "paced"
OTHER_KEY = "other"
CAP = 2
LOCK_HELD_TIMEOUT_SECONDS = 10


def _make_window(scope: str = SCOPE, cap: int = CAP) -> SharedRequestWindow:
    return SharedRequestWindow(scope, max_per_second={PACED_KEY: cap, OTHER_KEY: cap})


async def test_calls_within_the_cap_pass_without_waiting(virtual_sleeps):
    window = _make_window()

    await window.wait(PACED_KEY)
    await window.wait(PACED_KEY)

    virtual_sleeps.assert_not_called()


async def test_the_call_over_the_cap_waits_for_the_oldest_claim_to_leave(
    virtual_sleeps,
):
    window = _make_window()
    await window.wait(PACED_KEY)
    await window.wait(PACED_KEY)

    await window.wait(PACED_KEY)

    virtual_sleeps.assert_called_once_with(pytest.approx(1.0))


async def test_a_cancelled_wait_leaves_no_claim_behind(virtual_sleeps):
    window = _make_window(cap=1)
    await window.wait(PACED_KEY)
    cancelled_wait = asyncio.create_task(window.wait(PACED_KEY))
    await asyncio.sleep(0)
    cancelled_wait.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_wait

    stored_claims = json.loads(state_file_path(SCOPE).read_text())

    assert len(stored_claims[PACED_KEY]) == 1


async def test_a_key_without_a_cap_passes(virtual_sleeps):
    window = _make_window(cap=1)

    await window.wait("unbudgeted")
    await window.wait("unbudgeted")

    virtual_sleeps.assert_not_called()


async def test_keys_are_paced_apart(virtual_sleeps):
    window = _make_window(cap=1)

    await window.wait(PACED_KEY)
    await window.wait(OTHER_KEY)

    virtual_sleeps.assert_not_called()


async def test_windows_on_one_scope_share_their_claims(virtual_sleeps):
    await _make_window(cap=1).wait(PACED_KEY)

    await _make_window(cap=1).wait(PACED_KEY)

    virtual_sleeps.assert_called_once_with(pytest.approx(1.0))


async def test_scopes_are_isolated(virtual_sleeps):
    await _make_window(scope="scope-a", cap=1).wait(PACED_KEY)

    await _make_window(scope="scope-b", cap=1).wait(PACED_KEY)

    virtual_sleeps.assert_not_called()


@pytest.mark.parametrize(
    "unreadable_state",
    ["{not json", '{"paced": 3.0}', '{"paced": [null]}'],
)
async def test_corrupt_state_is_reset_to_an_open_window(
    unreadable_state, virtual_sleeps
):
    window = _make_window(cap=1)
    await window.wait(PACED_KEY)
    state_file_path(SCOPE).write_text(unreadable_state)

    await window.wait(PACED_KEY)

    virtual_sleeps.assert_not_called()


async def test_an_unreachable_budget_never_blocks(caplog, virtual_sleeps):
    state_file_path(SCOPE).parent.mkdir(parents=True, exist_ok=True)
    state_file_path(SCOPE).mkdir()
    window = _make_window(cap=1)

    with caplog.at_level(logging.WARNING):
        await window.wait(PACED_KEY)
        await window.wait(PACED_KEY)

    virtual_sleeps.assert_not_called()
    assert caplog.text.count("not shared between processes") == 1


async def test_a_held_lock_lets_the_caller_go(caplog):
    state_path = state_file_path(SCOPE)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    window = _make_window(cap=1)

    with open(state_path, "a+") as holder:
        await acquire_lock(holder)
        try:
            with caplog.at_level(logging.WARNING):
                async with asyncio.timeout(LOCK_HELD_TIMEOUT_SECONDS):
                    await window.wait(PACED_KEY)
        finally:
            release_lock(holder)

    assert "not shared between processes" in caplog.text
