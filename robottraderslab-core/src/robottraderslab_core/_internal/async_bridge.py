import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any


async def drain_tasks(tasks: list[asyncio.Task]) -> None:
    """Cancel and await the tasks so no exception is left unretrieved."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from synchronous code.

    Entry-point runner: uses a fresh event loop, falling back to a one-off
    thread when the calling thread already runs a loop (notebooks). Tasks on
    that loop start eagerly, running up to their first suspension as they are
    created, so one that never suspends is never scheduled.

    Args:
        coro: Coroutine to drive to completion.

    Returns:
        The coroutine's result.
    """
    if not _loop_is_already_running():
        return _run_on_eager_loop(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run_on_eager_loop, coro).result()


def _loop_is_already_running() -> bool:
    """Keep this check out of the caller's exception chain.

    Python chains whatever is raised inside an `except` block onto the error
    being handled, so asking this question inline attaches it to every failure
    the caller reports.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _run_on_eager_loop[T](coro: Coroutine[Any, Any, T]) -> T:
    with asyncio.Runner() as runner:
        runner.get_loop().set_task_factory(asyncio.eager_task_factory)
        return runner.run(coro)
