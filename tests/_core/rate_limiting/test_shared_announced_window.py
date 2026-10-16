import logging

import pytest

from robottraderslab._core.rate_limiting import state_file_path
from robottraderslab.exchanges import SharedAnnouncedWindow

SCOPE = "test-announced-scope"
KEY = "/api/market/candles"
OTHER_KEY = "/api/market/ticker"
DEFAULT_CAP = 2
FIRST_WINDOW = 100


def _make_window(scope: str = SCOPE) -> SharedAnnouncedWindow:
    return SharedAnnouncedWindow(scope, default_per_second=DEFAULT_CAP)


async def _claims_before_waiting(window: SharedAnnouncedWindow, virtual_sleeps) -> int:
    claims = 0
    while not virtual_sleeps.called:
        await window.wait(KEY)
        claims += 1
    return claims - 1


async def test_an_announced_higher_cap_replaces_the_default(virtual_sleeps):
    window = _make_window()
    await window.record(KEY, per_second=5, window=FIRST_WINDOW)

    claims = await _claims_before_waiting(window, virtual_sleeps)

    assert claims == 5


async def test_an_announced_lower_cap_of_a_later_window_replaces_a_higher_one(
    virtual_sleeps,
):
    window = _make_window()
    await window.record(KEY, per_second=5, window=FIRST_WINDOW)
    await window.record(KEY, per_second=1, window=FIRST_WINDOW + 1)

    claims = await _claims_before_waiting(window, virtual_sleeps)

    assert claims == 1


@pytest.mark.parametrize(("first_cap", "second_cap"), [(4, 3), (3, 4)])
async def test_the_largest_cap_of_a_window_stands_whatever_the_arrival_order(
    virtual_sleeps, first_cap, second_cap
):
    window = _make_window()
    await window.record(KEY, per_second=first_cap, window=FIRST_WINDOW)
    await window.record(KEY, per_second=second_cap, window=FIRST_WINDOW)

    claims = await _claims_before_waiting(window, virtual_sleeps)

    assert claims == 4


async def test_an_answer_of_an_earlier_window_changes_nothing(virtual_sleeps):
    window = _make_window()
    await window.record(KEY, per_second=4, window=FIRST_WINDOW + 1)
    await window.record(KEY, per_second=1, window=FIRST_WINDOW)

    claims = await _claims_before_waiting(window, virtual_sleeps)

    assert claims == 4


async def test_a_window_recording_an_earlier_window_leaves_what_another_recorded(
    virtual_sleeps,
):
    await _make_window().record(KEY, per_second=4, window=FIRST_WINDOW + 1)
    await _make_window().record(KEY, per_second=1, window=FIRST_WINDOW)

    claims = await _claims_before_waiting(_make_window(), virtual_sleeps)

    assert claims == 4


async def test_a_window_recording_a_lower_cap_of_the_same_window_leaves_what_another_recorded(
    virtual_sleeps,
):
    await _make_window().record(KEY, per_second=4, window=FIRST_WINDOW)
    await _make_window().record(KEY, per_second=1, window=FIRST_WINDOW)

    claims = await _claims_before_waiting(_make_window(), virtual_sleeps)

    assert claims == 4


async def test_an_unreachable_state_never_blocks(caplog, virtual_sleeps):
    state_file_path(SCOPE).parent.mkdir(parents=True, exist_ok=True)
    state_file_path(SCOPE).mkdir()
    window = _make_window()

    with caplog.at_level(logging.WARNING):
        for _ in range(DEFAULT_CAP + 1):
            await window.wait(KEY)

    virtual_sleeps.assert_not_called()
    assert caplog.text.count("not shared between processes") == 1


async def test_a_window_started_later_paces_from_the_last_announcement(
    virtual_sleeps,
):
    await _make_window().record(KEY, per_second=5, window=FIRST_WINDOW)

    claims = await _claims_before_waiting(_make_window(), virtual_sleeps)

    assert claims == 5


async def test_an_announcement_moves_only_its_own_key(virtual_sleeps):
    window = _make_window()
    await window.record(OTHER_KEY, per_second=5, window=FIRST_WINDOW)

    claims = await _claims_before_waiting(window, virtual_sleeps)

    assert claims == DEFAULT_CAP


async def test_a_corrupt_state_resets_to_the_default(virtual_sleeps):
    window = _make_window()
    await window.wait(KEY)

    state_file_path(SCOPE).write_text("[1, 2")

    claims = await _claims_before_waiting(window, virtual_sleeps)

    assert claims == DEFAULT_CAP


async def test_the_call_over_the_cap_waits_for_the_oldest_claim_to_leave(
    virtual_sleeps,
):
    window = _make_window()
    for _ in range(DEFAULT_CAP):
        await window.wait(KEY)

    await window.wait(KEY)

    virtual_sleeps.assert_called_once_with(pytest.approx(1.0))
