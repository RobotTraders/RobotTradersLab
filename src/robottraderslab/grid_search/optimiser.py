import copy
import itertools
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any, cast

from robottraderslab.backtester import run_backtest
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import StrategyCriticalError

from .grid_point import GridPoint
from .optimisation_config import OptimisationConfig
from .optimisation_result import OptimisationResult
from .parameter import Parameter, ParameterValue
from .pooled_sweep import log_outcome, measure_pooled

logger = logging.getLogger(__name__)

REQUIRED_PARAMETER_COUNT = 2
_BACKTESTER_LOGGER_NAME = "robottraderslab.backtester"


class GridSearchOptimiser:
    """Orchestrates grid search optimisation over strategy parameters."""

    def __init__(self, bot_config: BotConfig) -> None:
        self._bot_config = bot_config
        self._optimisation_config = OptimisationConfig.from_config(
            _require_optimisation_config(bot_config)
        )

    def run(self) -> list[OptimisationResult]:
        parameters = self._optimisation_config.to_parameters(self._current_values())
        grid_points = _generate_grid(parameters)
        max_workers = self._optimisation_config.max_workers

        if max_workers == 1:
            return self._run_sequential(grid_points)
        return self._run_parallel(grid_points, max_workers)

    def _current_values(self) -> dict[str, ParameterValue]:
        config_dict = self._bot_config.model_dump()
        return {
            declared.parameter: _read_nested_value(config_dict, declared.parameter)
            for declared in self._optimisation_config.parameter_configs
        }

    def _run_parallel(
        self, grid_points: list[GridPoint], max_workers: int
    ) -> list[OptimisationResult]:
        """A range samples from `min` to `max`, so the last point holds the
        largest values: its backtest, run before the pool starts, fetches the
        longest lookback a sweep of lengths declares, and the workers find
        those candles on disk. A point declaring a longer lookback still
        fetches the difference itself.
        """
        total_points = len(grid_points)
        config_dict = self._bot_config.model_dump()

        logger.info(
            f"Starting parallel grid search: {total_points} points, "
            f"max_workers={max_workers}"
        )

        last = total_points - 1
        warm_up = _measure_point(config_dict, grid_points[last])
        log_outcome(warm_up, last + 1, total_points)
        results_by_index = measure_pooled(
            partial(_measure_point, config_dict),
            dict(enumerate(grid_points[:last])),
            max_workers,
            total_points,
        )
        results_by_index[last] = warm_up

        logger.info(f"Grid search completed. {total_points} results")
        return [results_by_index[i] for i in range(total_points)]

    def _run_sequential(self, grid_points: list[GridPoint]) -> list[OptimisationResult]:
        total_points = len(grid_points)
        config_dict = self._bot_config.model_dump()
        results: list[OptimisationResult] = []

        for i, grid_point in enumerate(grid_points, start=1):
            logger.info(f"Evaluating grid point {i}/{total_points}: {grid_point}")

            result = _measure_point(config_dict, grid_point)
            results.append(result)

            if not result.is_successful:
                logger.warning(f"Grid point {i}/{total_points} failed: {result.error}")

        logger.info(f"Grid search completed. {len(results)} results")
        return results


def _generate_grid(parameters: list[Parameter]) -> list[GridPoint]:
    """Pair every value of one parameter with every value of the other.

    A grid is drawn in two dimensions, so a sweep declares two parameters.

    Returns:
        List of GridPoint objects, one for each combination.

    Raises:
        StrategyCriticalError: If the config declares any other count.
    """
    if len(parameters) != REQUIRED_PARAMETER_COUNT:
        raise StrategyCriticalError(_wrong_count(len(parameters)))

    param1, param2 = parameters

    values1 = param1.generate_values()
    values2 = param2.generate_values()

    grid_points = []
    for val1, val2 in itertools.product(values1, values2):
        grid_points.append(
            GridPoint(
                parameters={
                    param1.name: val1,
                    param2.name: val2,
                }
            )
        )

    return grid_points


def _wrong_count(declared: int) -> str:
    """Only a config declaring too many has parameters left over to place."""
    refused = (
        f"A sweep declares exactly {REQUIRED_PARAMETER_COUNT} parameters and "
        f"`[optimisation]` declares {declared}. No other count is supported"
    )
    if declared < REQUIRED_PARAMETER_COUNT:
        return refused
    return f"{refused}; the rest stay under `[strategy]` at the value they hold"


def _measure_point(
    config_dict: dict[str, Any],
    grid_point: GridPoint,
) -> OptimisationResult:
    """Measure one point apart from the sweep, so a failure costs that point alone."""
    try:
        bot_config = _apply_grid_point_to_config(copy.deepcopy(config_dict), grid_point)
        with _silenced_backtest_logs():
            backtest_outputs = run_backtest(bot_config)
        return OptimisationResult.from_backtest(grid_point, backtest_outputs)

    except Exception as e:
        logger.exception("Error evaluating grid point %s", grid_point)
        return OptimisationResult.from_error(grid_point, str(e))


def _apply_grid_point_to_config(
    config_dict: dict[str, Any], grid_point: GridPoint
) -> BotConfig:
    """Patch one point's values into a config and validate what comes out.

    Args:
        config_dict: Mutable config dictionary to modify in place.

    Returns:
        The bot config a single point is measured by.
    """
    for parameter_path, value in grid_point.parameters.items():
        _set_nested_value(config_dict, parameter_path, value)

    return BotConfig.model_validate(config_dict)


@contextmanager
def _silenced_backtest_logs() -> Iterator[None]:
    backtest_logger = logging.getLogger(_BACKTESTER_LOGGER_NAME)
    previous_level = backtest_logger.level
    backtest_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        backtest_logger.setLevel(previous_level)


def _set_nested_value(
    config_dict: dict[str, Any], path: str, value: ParameterValue
) -> None:
    """A path is checked against the config before any point is measured on it.

    Raises:
        StrategyCriticalError: If a segment names nothing the config declares.
    """
    _assign(config_dict, path.split("."), value, path)


def _assign(target: Any, segments: list[str], value: ParameterValue, path: str) -> None:
    if isinstance(target, list):
        _assign_in_list(target, segments, value, path)
    elif isinstance(target, dict):
        _assign_in_table(target, segments, value, path)
    else:
        raise _under_a_single_value(path, segments[0])


def _assign_in_list(
    target: list[Any], segments: list[str], value: ParameterValue, path: str
) -> None:
    segment, remaining = segments[0], segments[1:]

    if not segment.isdigit():
        for item in target:
            _assign(item, segments, value, path)
        return

    index = int(segment)
    if index >= len(target):
        raise _past_the_end(path, index, len(target))

    if remaining:
        _assign(target[index], remaining, value, path)
    else:
        target[index] = value


def _assign_in_table(
    target: dict[str, Any], segments: list[str], value: ParameterValue, path: str
) -> None:
    segment, remaining = segments[0], segments[1:]

    if segment not in target:
        raise _undeclared(path, segment)

    if remaining:
        _assign(target[segment], remaining, value, path)
    else:
        target[segment] = value


def _read_nested_value(config_dict: dict[str, Any], path: str) -> ParameterValue:
    """A path is read before any point is measured, so a setting the config
    does not spell out stops the sweep before a backtest runs.

    Raises:
        StrategyCriticalError: If a segment names nothing the config declares,
            or the path lands on a table, a list or an unset value.
    """
    value = _read(config_dict, path.split("."), path)
    if isinstance(value, dict | list):
        raise StrategyCriticalError(
            f"Optimisation parameter `{path}` names a table or a list, which "
            f"holds no single value to sweep"
        )
    if value is None:
        raise StrategyCriticalError(
            f"Optimisation parameter `{path}` names a setting the config leaves "
            f"unset, so its type cannot be read"
        )
    return cast(ParameterValue, value)


def _read(target: Any, segments: list[str], path: str) -> Any:
    """A list is read through its first item, the one every item stands for
    when a path fans out over them.
    """
    if not segments:
        return target
    segment, remaining = segments[0], segments[1:]
    if isinstance(target, list):
        if not segment.isdigit():
            return _read(_first_of(target, path), segments, path)
        index = int(segment)
        if index >= len(target):
            raise _past_the_end(path, index, len(target))
        return _read(target[index], remaining, path)
    if isinstance(target, dict):
        if segment not in target:
            raise _undeclared(path, segment)
        return _read(target[segment], remaining, path)
    raise _under_a_single_value(path, segment)


def _first_of(target: list[Any], path: str) -> Any:
    if not target:
        raise StrategyCriticalError(
            f"Optimisation parameter `{path}` fans out over an empty list, "
            f"which holds nothing to sweep"
        )
    return target[0]


def _undeclared(path: str, segment: str) -> StrategyCriticalError:
    return StrategyCriticalError(
        f"Optimisation parameter `{path}` names `{segment}`, which the "
        f"config does not declare"
    )


def _past_the_end(path: str, index: int, length: int) -> StrategyCriticalError:
    return StrategyCriticalError(
        f"Optimisation parameter `{path}` names element {index} of a list "
        f"holding {length}"
    )


def _under_a_single_value(path: str, segment: str) -> StrategyCriticalError:
    return StrategyCriticalError(
        f"Optimisation parameter `{path}` reaches `{segment}` under a "
        f"single value, which holds nothing to address"
    )


def _require_optimisation_config(bot_config: BotConfig) -> dict[str, Any]:
    """Read the section a sweep cannot run without.

    Raises:
        StrategyCriticalError: If the bot declares no `[optimisation]` section.
    """
    if bot_config.optimisation is None:
        raise StrategyCriticalError(
            "`[optimisation]` section is required to run a grid search"
        )
    return bot_config.optimisation
