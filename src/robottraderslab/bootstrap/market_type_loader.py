import logging
from typing import Any, cast

from robottraderslab._core import ClassLoadingError, load_class
from robottraderslab.exceptions import StrategyCriticalError

from .market_type import SimulatedMarket

logger = logging.getLogger(__name__)

_MARKET_TYPE_ENTRY_POINT_GROUP = "robot_traders_lab.market_types"


def load_simulated_market(market_type: str, **settings: Any) -> SimulatedMarket:
    """Build the simulated market a strategy's market type maps to.

    Args:
        market_type: Market type declared by the strategy, resolved through the
            ``robot_traders_lab.market_types`` entry point group.
        **settings: Forwarded to the registered factory (initial balance, fees).

    Returns:
        The simulator and account for this market type.

    Raises:
        StrategyCriticalError: If no installed package registers the market type.
    """
    try:
        factory = load_class(market_type, _MARKET_TYPE_ENTRY_POINT_GROUP)
    except ClassLoadingError as e:
        raise StrategyCriticalError(
            f"Market type `{market_type}` is not installed; "
            "install the package that provides it or check the strategy's `market_type`"
        ) from e
    logger.debug(f"Market type `{market_type}` resolved")
    return cast(SimulatedMarket, factory(**settings))
