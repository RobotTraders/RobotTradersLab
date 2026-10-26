import asyncio
import logging
import os
import stat
import sys
from collections.abc import Awaitable, Callable

import pytest

from robottraderslab._core.rate_limiting import (
    SharedAnnouncedWindow,
    SharedRequestWindow,
    SharedTokenBucket,
    SharedVenueWindow,
    acquire_lock,
    release_lock,
    state_file_path,
)
from robottraderslab._core.rate_limiting.file_lock import LOCK_WAIT_LIMIT_SECONDS

SCOPE = "test-scope"
POLL_ALLOWANCE_SECONDS = 0.05
RELEASE_TIMEOUT_SECONDS = 2.0
KEY = "budget"
CAP = 1
HOLD_SECONDS = 1.0
LOCK_WAIT_MARGIN = 3
ANOTHER_USER_UID = 99999
PRIVATE_FILE_MODE = 0o600
LINK_TARGET_CONTENT = "not a budget"

type Claim = Callable[[], Awaitable[None]]

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="%TEMP% is the user's own on Windows"
)


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
            acquire_lock(waiting), timeout=LOCK_WAIT_LIMIT_SECONDS * LOCK_WAIT_MARGIN
        )
        release_lock(held)

    assert not took_the_lock


def _request_window_claim(scope: str) -> Claim:
    window = SharedRequestWindow(scope, max_per_second={KEY: CAP})
    return lambda: window.wait(KEY)


def _announced_window_claim(scope: str) -> Claim:
    window = SharedAnnouncedWindow(scope, default_per_second=CAP)
    return lambda: window.wait(KEY)


def _token_bucket_claim(scope: str) -> Claim:
    bucket = SharedTokenBucket(scope, capacity=CAP, refill_per_second=CAP)
    return lambda: bucket.consume(CAP)


def _venue_window_claim(scope: str) -> Claim:
    window = SharedVenueWindow(scope)

    async def hold_then_wait() -> None:
        await window.hold(KEY, HOLD_SECONDS)
        await window.wait(KEY)

    return hold_then_wait


CLAIMS = [
    pytest.param(_request_window_claim, id="request window"),
    pytest.param(_announced_window_claim, id="announced window"),
    pytest.param(_token_bucket_claim, id="token bucket"),
    pytest.param(_venue_window_claim, id="venue window"),
]


@posix_only
async def test_the_state_file_is_readable_and_writable_by_its_user_alone():
    await _request_window_claim(SCOPE)()

    assert stat.S_IMODE(state_file_path(SCOPE).stat().st_mode) == PRIVATE_FILE_MODE


@posix_only
@pytest.mark.parametrize("make_claim", CLAIMS)
async def test_a_state_directory_another_user_owns_stays_empty(
    make_claim, caplog, monkeypatch, make_unshared_report_levels, virtual_sleeps
):
    state_dir = state_file_path(SCOPE).parent
    state_dir.mkdir(parents=True)
    another_user = ANOTHER_USER_UID
    monkeypatch.setattr(os, "geteuid", lambda: another_user)
    claim = make_claim(SCOPE)

    with caplog.at_level(logging.WARNING):
        await claim()
        await claim()

    assert list(state_dir.iterdir()) == []
    assert make_unshared_report_levels() == ["ERROR"]
    virtual_sleeps.assert_not_called()


@posix_only
@pytest.mark.parametrize("make_claim", CLAIMS)
async def test_a_link_at_the_state_path_leaves_its_target_untouched(
    make_claim, caplog, tmp_path, make_unshared_report_levels, virtual_sleeps
):
    link_target = tmp_path / "link-target"
    link_target.write_text(LINK_TARGET_CONTENT)
    state_path = state_file_path(SCOPE)
    state_path.parent.mkdir(parents=True)
    state_path.symlink_to(link_target)
    claim = make_claim(SCOPE)

    with caplog.at_level(logging.WARNING):
        await claim()
        await claim()

    assert link_target.read_text() == LINK_TARGET_CONTENT
    assert make_unshared_report_levels() == ["ERROR"]
    virtual_sleeps.assert_not_called()
