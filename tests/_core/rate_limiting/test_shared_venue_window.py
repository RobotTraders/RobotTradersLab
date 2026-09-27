import asyncio
import json
import logging
from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest

from robottraderslab._core.rate_limiting import (
    acquire_lock,
    release_lock,
    state_file_path,
)
from robottraderslab.exchanges import SharedVenueWindow

SCOPE = "test-venue-scope"
KEY = "webhook"
WINDOW = "1470173023"
NEXT_WINDOW = "1470173024"
RESET_AFTER = 1.0
HOLD_SECONDS = 5.0
LOCK_HELD_TIMEOUT_SECONDS = 10
MONOTONIC_BEHIND_SECONDS = 100_000.0

_unpatched_sleep = asyncio.sleep


@pytest.fixture
def window() -> SharedVenueWindow:
    return SharedVenueWindow(SCOPE)


@pytest.fixture
def clocks_of_a_fresh_boot(clock) -> Iterator[AsyncMock]:
    """A monotonic reading counts from the boot it was taken in, so after a
    reboot it sits far behind the wall clock a state file was written on.
    """

    async def advance(delay: float) -> None:
        clock.now += delay
        await _unpatched_sleep(0)

    with (
        patch("time.time", side_effect=clock),
        patch(
            "time.monotonic", side_effect=lambda: clock.now - MONOTONIC_BEHIND_SECONDS
        ),
        patch(
            "asyncio.sleep", new_callable=AsyncMock, side_effect=advance
        ) as sleep_mock,
    ):
        yield sleep_mock


def _spend_the_key_until(deadline: float) -> None:
    state_path = state_file_path(SCOPE)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({KEY: {"remaining": 0, "window": WINDOW, "reset_at": deadline}})
    )


async def test_the_announced_count_bounds_the_claims_before_the_reset(
    window, virtual_sleeps
):
    await window.record(KEY, remaining=2, window=WINDOW, reset_after=RESET_AFTER)

    await window.wait(KEY)
    await window.wait(KEY)

    virtual_sleeps.assert_not_called()


async def test_a_claim_on_a_spent_count_waits_for_the_announced_reset(
    window, virtual_sleeps
):
    await window.record(KEY, remaining=1, window=WINDOW, reset_after=RESET_AFTER)
    await window.wait(KEY)

    await window.wait(KEY)

    virtual_sleeps.assert_called_once_with(pytest.approx(RESET_AFTER))


async def test_a_key_the_venue_has_not_answered_for_passes(window, virtual_sleeps):
    await window.wait(KEY)
    await window.wait(KEY)

    virtual_sleeps.assert_not_called()


async def test_a_second_answer_about_one_window_never_hands_a_claim_back(
    window, virtual_sleeps
):
    await window.record(KEY, remaining=2, window=WINDOW, reset_after=RESET_AFTER)
    await window.wait(KEY)
    await window.record(KEY, remaining=2, window=WINDOW, reset_after=RESET_AFTER)

    await window.wait(KEY)
    await window.wait(KEY)

    virtual_sleeps.assert_called_once_with(pytest.approx(RESET_AFTER))


async def test_an_answer_about_a_new_window_replaces_the_one_before_it(
    window, virtual_sleeps
):
    await window.record(KEY, remaining=2, window=WINDOW, reset_after=RESET_AFTER)
    await window.wait(KEY)
    await window.record(KEY, remaining=2, window=NEXT_WINDOW, reset_after=RESET_AFTER)

    await window.wait(KEY)
    await window.wait(KEY)

    virtual_sleeps.assert_not_called()


async def test_a_hold_stands_through_an_answer_sent_before_it(window, virtual_sleeps):
    await window.hold(KEY, HOLD_SECONDS)
    await window.record(KEY, remaining=2, window=WINDOW, reset_after=RESET_AFTER)

    await window.wait(KEY)

    virtual_sleeps.assert_called_once_with(pytest.approx(HOLD_SECONDS))


async def test_a_hold_bars_a_key_the_venue_has_not_answered_for(window, virtual_sleeps):
    await window.hold(KEY, HOLD_SECONDS)

    await window.wait(KEY)

    virtual_sleeps.assert_called_once_with(pytest.approx(HOLD_SECONDS))


async def test_keys_are_paced_apart(window, virtual_sleeps):
    await window.record(KEY, remaining=1, window=WINDOW, reset_after=RESET_AFTER)
    await window.wait(KEY)

    await window.wait("another-webhook")

    virtual_sleeps.assert_not_called()


async def test_windows_on_one_scope_draw_on_one_another(virtual_sleeps):
    await SharedVenueWindow(SCOPE).record(
        KEY, remaining=1, window=WINDOW, reset_after=RESET_AFTER
    )
    await SharedVenueWindow(SCOPE).wait(KEY)

    await SharedVenueWindow(SCOPE).wait(KEY)

    virtual_sleeps.assert_called_once_with(pytest.approx(RESET_AFTER))


async def test_scopes_are_isolated(virtual_sleeps):
    await SharedVenueWindow("scope-a").record(
        KEY, remaining=1, window=WINDOW, reset_after=RESET_AFTER
    )
    await SharedVenueWindow("scope-a").wait(KEY)

    await SharedVenueWindow("scope-b").wait(KEY)

    virtual_sleeps.assert_not_called()


@pytest.mark.parametrize(
    "unreadable_state",
    [
        "{not json",
        '{"webhook": 3.0}',
        '{"webhook": {"remaining": null, "window": "1", "reset_at": 2.0}}',
        '{"webhook": {"window": "1", "reset_at": 2.0}}',
    ],
)
async def test_corrupt_state_is_reset_to_an_unpaced_key(
    unreadable_state, window, virtual_sleeps
):
    await window.record(KEY, remaining=0, window=WINDOW, reset_after=HOLD_SECONDS)
    state_file_path(SCOPE).write_text(unreadable_state)

    await window.wait(KEY)

    virtual_sleeps.assert_not_called()


async def test_an_unreachable_budget_never_blocks(window, caplog, virtual_sleeps):
    state_file_path(SCOPE).parent.mkdir(parents=True, exist_ok=True)
    state_file_path(SCOPE).mkdir()

    with caplog.at_level(logging.WARNING):
        await window.hold(KEY, HOLD_SECONDS)
        await window.wait(KEY)

    virtual_sleeps.assert_not_called()
    assert caplog.text.count("not shared between processes") == 1


async def test_a_deadline_a_previous_boot_recorded_has_passed(
    window, clock, clocks_of_a_fresh_boot
):
    _spend_the_key_until(clock.now - RESET_AFTER)

    await window.wait(KEY)

    clocks_of_a_fresh_boot.assert_not_called()


async def test_a_deadline_that_has_passed_lets_the_next_answer_through(
    window, clock, clocks_of_a_fresh_boot
):
    _spend_the_key_until(clock.now - RESET_AFTER)

    await window.record(KEY, remaining=2, window=NEXT_WINDOW, reset_after=RESET_AFTER)
    await window.wait(KEY)
    await window.wait(KEY)

    clocks_of_a_fresh_boot.assert_not_called()


async def test_a_held_lock_lets_the_caller_go(window, caplog):
    state_path = state_file_path(SCOPE)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    with open(state_path, "a+") as holder:
        await acquire_lock(holder)
        try:
            with caplog.at_level(logging.WARNING):
                async with asyncio.timeout(LOCK_HELD_TIMEOUT_SECONDS):
                    await window.wait(KEY)
        finally:
            release_lock(holder)

    assert "not shared between processes" in caplog.text
