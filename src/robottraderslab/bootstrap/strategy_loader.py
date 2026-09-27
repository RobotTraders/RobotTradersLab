import logging
from collections.abc import Callable
from typing import Any

from robottraderslab._core import Candles, ChartLine, ClassLoadingError, load_class
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.strategies import StrategyProtocol

logger = logging.getLogger(__name__)

_Section = dict[str, Any]

_STRATEGY_CLASS_KEY = "strategy_class"
_STRATEGY_ENTRY_POINT_GROUP = "robot_traders_lab.strategies"
_INDICATORS_ENTRY_POINT_GROUP = "robot_traders_lab.indicators"

ChartLinesFunc = Callable[[Candles, dict[str, Any]], list[ChartLine]]


def load_strategy(strategy_config: _Section, **auto_params: Any) -> StrategyProtocol:
    """Load the strategy a config names, with its own parameters applied.

    Args:
        strategy_config: must contain a `strategy_class` key specifying either:
          - A short name registered via entry points (e.g., "impulse")
          - A full class path (e.g., "robottraderslab_impulse.impulse.ImpulseStrategy")
    """
    strategy_config.update(**auto_params)
    strategy_name = _extract_strategy_name(strategy_config)
    strategy_class = _load_strategy_class(strategy_name)
    instance = strategy_class(**strategy_config)
    strategy = _validate_strategy(strategy_name, instance)
    logger.debug(f"Strategy `{strategy_name}` loaded")
    return strategy


def load_lightweight_chart_indicators(
    indicators_name: str,
    candles: Candles,
    indicators_params: dict[str, Any],
) -> list[ChartLine]:
    """Load a strategy's chart lines by entry point name or dotted path.

    Args:
        indicators_name: Entry point name, or the full path of a function.

    Raises:
        StrategyCriticalError: If the lines cannot be loaded.
    """
    try:
        chart_lines: ChartLinesFunc = load_class(
            indicators_name, _INDICATORS_ENTRY_POINT_GROUP
        )
        return chart_lines(candles, indicators_params)
    except ClassLoadingError as e:
        raise StrategyCriticalError(str(e)) from e


def _extract_strategy_name(config: _Section) -> str:
    try:
        name: str = config.pop(_STRATEGY_CLASS_KEY)
        return name
    except KeyError:
        raise StrategyCriticalError(f"`{_STRATEGY_CLASS_KEY}` is a required config key")


def _load_strategy_class(strategy_name: str) -> Any:
    try:
        return load_class(strategy_name, _STRATEGY_ENTRY_POINT_GROUP)
    except ClassLoadingError as e:
        raise StrategyCriticalError(str(e)) from e


def _validate_strategy(strategy_class_name: str, instance: Any) -> StrategyProtocol:
    if not isinstance(instance, StrategyProtocol):
        raise TypeError(
            f"Make sure your strategy class `{strategy_class_name}` implements `StrategyProtocol` accordingly"
        )
    return instance
