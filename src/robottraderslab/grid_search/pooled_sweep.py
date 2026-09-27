import logging
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack

from robottraderslab.exceptions import StrategyCriticalError

from .grid_point import GridPoint
from .optimisation_result import OptimisationResult

logger = logging.getLogger(__name__)

MAX_POOL_BREAKS = 3

type _Measure = Callable[[GridPoint], OptimisationResult]


def measure_pooled(
    measure: _Measure,
    grid_points: Mapping[int, GridPoint],
    max_workers: int,
    total_points: int,
) -> dict[int, OptimisationResult]:
    """A worker killed from outside costs the sweep at most the point it was running.

    Args:
        measure: Picklable, since each point crosses into a worker process.
        grid_points: By their position in the grid.
        max_workers: The size of every pool the sweep opens, before a break
            and after it.
        total_points: The grid's size, which each logged outcome is counted
            against.

    Raises:
        StrategyCriticalError: If the pool breaks more than `MAX_POOL_BREAKS`
            times.
    """
    return _PooledSweep(measure, grid_points, max_workers, total_points).run()


def log_outcome(result: OptimisationResult, position: int, total_points: int) -> None:
    if result.is_successful:
        logger.info(f"Grid point {position}/{total_points} completed")
    else:
        logger.warning(f"Grid point {position}/{total_points} failed: {result.error}")


class _PooledSweep:
    def __init__(
        self,
        measure: _Measure,
        grid_points: Mapping[int, GridPoint],
        max_workers: int,
        total_points: int,
    ) -> None:
        self._measure = measure
        self._grid_points = grid_points
        self._max_workers = max_workers
        self._total_points = total_points
        self._results: dict[int, OptimisationResult] = {}

    def run(self) -> dict[int, OptimisationResult]:
        pool_breaks = 0
        while True:
            running_at_break = self._measure_until_broken()
            if len(self._results) == len(self._grid_points):
                return self._results
            pool_breaks += 1
            if pool_breaks > MAX_POOL_BREAKS:
                raise StrategyCriticalError(_too_many_breaks(pool_breaks))
            logger.warning(
                f"The process pool broke ({pool_breaks}/{MAX_POOL_BREAKS}): "
                f"{len(self._results)} results kept, "
                f"{len(self._grid_points) - len(self._results)} points left to measure"
            )
            self._measure_each_alone(running_at_break)

    def _collect(
        self,
        done: Iterable[Future[OptimisationResult]],
        in_flight: dict[Future[OptimisationResult], int],
    ) -> set[int]:
        lost: set[int] = set()
        for future in done:
            i = in_flight.pop(future)
            if isinstance(future.exception(), BrokenProcessPool):
                lost.add(i)
            else:
                self._record(i, _outcome_of(future, self._grid_points[i]))
        return lost

    def _collect_after_break(
        self, in_flight: dict[Future[OptimisationResult], int]
    ) -> set[int]:
        """A broken pool fails its futures one by one."""
        wait(in_flight)
        return self._collect(list(in_flight), in_flight)

    def _measure_each_alone(self, indices: Iterable[int]) -> None:
        """A pool per point, so a kill fails only the point that drew it."""
        with ExitStack() as pools:
            futures = {
                i: pools.enter_context(ProcessPoolExecutor(max_workers=1)).submit(
                    self._measure, self._grid_points[i]
                )
                for i in sorted(indices)
            }
            for i, future in futures.items():
                self._record(i, _outcome_of(future, self._grid_points[i]))

    def _measure_until_broken(self) -> set[int]:
        """Only a point a worker is running may be in flight when the pool
        breaks.
        """
        queued = deque(i for i in self._grid_points if i not in self._results)
        in_flight: dict[Future[OptimisationResult], int] = {}
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            while queued or in_flight:
                try:
                    while queued and len(in_flight) < self._max_workers:
                        i = queued.popleft()
                        point = self._grid_points[i]
                        in_flight[executor.submit(self._measure, point)] = i
                except BrokenProcessPool as e:
                    logger.warning(f"The process pool refused point {i + 1}: {e}")
                    return self._collect_after_break(in_flight)
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                lost = self._collect(done, in_flight)
                if lost:
                    return lost | self._collect_after_break(in_flight)
        return set()

    def _record(self, i: int, result: OptimisationResult) -> None:
        self._results[i] = result
        log_outcome(result, i + 1, self._total_points)


def _too_many_breaks(pool_breaks: int) -> str:
    return (
        f"The process pool broke {pool_breaks} times, more than the "
        f"{MAX_POOL_BREAKS} a sweep rides out: something outside the engine keeps "
        f"killing its workers"
    )


def _outcome_of(
    future: Future[OptimisationResult], grid_point: GridPoint
) -> OptimisationResult:
    """A critical is no `Exception`, so it crosses the pool carrying its class
    and stops the sweep.
    """
    try:
        return future.result()
    except BrokenProcessPool as e:
        logger.warning("Grid point %s lost its worker: %s", grid_point, e)
        return OptimisationResult.from_error(grid_point, str(e))
    except Exception as e:
        logger.exception("Grid point %s returned no result", grid_point)
        return OptimisationResult.from_error(grid_point, str(e))
