import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.grid_search.grid_point import GridPoint
from robottraderslab.grid_search.optimisation_result import OptimisationResult
from robottraderslab.grid_search.pooled_sweep import MAX_POOL_BREAKS, measure_pooled

GRID_SIZE = 6
KILLER = 3


@dataclass(frozen=True)
class KillsItsWorker:
    """Counts attempts in files, since each one runs in a different process."""

    attempts_dir: Path
    killers: frozenset[int]
    lethal_attempts: int

    def __call__(self, grid_point: GridPoint) -> OptimisationResult:
        position = int(grid_point.parameters["position"])
        attempt = len(list(self.attempts_dir.glob(f"{position}-*"))) + 1
        (self.attempts_dir / f"{position}-{attempt}").touch()
        if position in self.killers and attempt <= self.lethal_attempts:
            os._exit(1)
        return OptimisationResult(grid_point=grid_point, roi=float(position))


@dataclass(frozen=True)
class RaisesACritical:
    critical_point: int

    def __call__(self, grid_point: GridPoint) -> OptimisationResult:
        if grid_point.parameters["position"] == self.critical_point:
            raise StrategyCriticalError("the strategy declares no candles")
        return OptimisationResult(grid_point=grid_point)


@dataclass(frozen=True)
class ReturnsAnUnpicklableResult:
    def __call__(self, grid_point: GridPoint) -> OptimisationResult:
        return OptimisationResult(grid_point=GridPoint(parameters={"x": lambda: 0}))


def _grid(size: int) -> dict[int, GridPoint]:
    return {i: GridPoint(parameters={"position": i}) for i in range(size)}


def test_a_worker_killed_once_costs_no_point(tmp_path):
    kills_once = KillsItsWorker(tmp_path, frozenset({KILLER}), lethal_attempts=1)

    results = measure_pooled(kills_once, _grid(GRID_SIZE), 2, GRID_SIZE)

    assert sorted(results) == list(range(GRID_SIZE))
    assert [results[i].roi for i in range(GRID_SIZE)] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_a_point_killing_its_worker_twice_fails_alone(tmp_path):
    kills_twice = KillsItsWorker(tmp_path, frozenset({KILLER}), lethal_attempts=2)

    results = measure_pooled(kills_twice, _grid(GRID_SIZE), 2, GRID_SIZE)

    assert "terminated abruptly" in results[KILLER].error
    assert [results[i].roi for i in range(GRID_SIZE) if i != KILLER] == [
        0.0,
        1.0,
        2.0,
        4.0,
        5.0,
    ]


def test_a_pool_breaking_past_the_cap_stops_the_sweep(tmp_path):
    breaks = MAX_POOL_BREAKS + 1
    kills_each_once = KillsItsWorker(
        tmp_path, frozenset(range(breaks)), lethal_attempts=1
    )

    with pytest.raises(StrategyCriticalError, match=f"broke {breaks} times"):
        measure_pooled(kills_each_once, _grid(breaks), 1, breaks)


def test_a_critical_raised_in_a_worker_stops_the_sweep():
    with pytest.raises(StrategyCriticalError, match="declares no candles"):
        measure_pooled(RaisesACritical(KILLER), _grid(GRID_SIZE), 2, GRID_SIZE)


def test_a_result_that_cannot_travel_back_fails_its_point():
    results = measure_pooled(ReturnsAnUnpicklableResult(), _grid(1), 1, 1)

    assert not results[0].is_successful
    assert results[0].parameters == {"position": 0}
