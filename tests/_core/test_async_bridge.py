import asyncio

import pytest

from robottraderslab._core import run_async
from robottraderslab.exceptions import StrategyCriticalError


async def _loop_identity() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


async def _refuses() -> None:
    raise ValueError("what the caller actually needs to read")


async def _stops() -> None:
    raise StrategyCriticalError("what the caller actually needs to read")


class TestRunAsync:
    def test_runs_coroutine_without_a_loop(self):
        used_loop = run_async(_loop_identity())

        assert isinstance(used_loop, asyncio.AbstractEventLoop)

    async def test_runs_coroutine_from_a_running_loop(self):
        outer_loop = asyncio.get_running_loop()

        inner_loop = run_async(_loop_identity())

        assert inner_loop is not outer_loop

    def test_a_coroutine_that_raises(self):
        with pytest.raises(
            ValueError, match="what the caller actually needs"
        ) as failure:
            run_async(_refuses())

        assert failure.value.__context__ is None

    def test_a_coroutine_that_raises_a_critical(self):
        with pytest.raises(StrategyCriticalError, match="what the caller actually"):
            run_async(_stops())

    async def test_a_coroutine_that_raises_a_critical_from_a_running_loop(self):
        with pytest.raises(StrategyCriticalError, match="what the caller actually"):
            run_async(_stops())
