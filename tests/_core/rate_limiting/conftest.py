import asyncio
import tempfile
from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest

START_TIME = 1_000.0

_unpatched_sleep = asyncio.sleep


@pytest.fixture(autouse=True)
def _temporary_state_dir(tmp_path, monkeypatch) -> None:
    """`state_file_path` resolves the temporary directory on every call, so a
    scope written by one test is unreachable from the next.
    """
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))


class Clock:
    """Time moves only when a test or a sleep moves it, so a delay a limiter
    asks for never elapses on its own.
    """

    def __init__(self) -> None:
        self.now = START_TIME

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def virtual_sleeps(clock: Clock) -> Iterator[AsyncMock]:
    """One clock behind `time.time` and `asyncio.sleep`: a sleep advances it
    by the delay and hands the loop back, so a limiter that re-checks on
    waking finds the moment it asked for without the test taking that long.
    """

    async def advance(delay: float) -> None:
        clock.now += delay
        await _unpatched_sleep(0)

    with (
        patch("time.time", side_effect=clock),
        patch("time.monotonic", side_effect=clock),
        patch(
            "asyncio.sleep", new_callable=AsyncMock, side_effect=advance
        ) as sleep_mock,
    ):
        yield sleep_mock
