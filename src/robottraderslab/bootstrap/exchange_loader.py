import inspect
import logging
from collections.abc import Mapping
from typing import Any, cast

from robottraderslab._core import (
    ClassLoadingError,
    load_class,
    settings_the_factory_does_not_take,
)
from robottraderslab.exceptions import ExchangeCriticalError
from robottraderslab.exchanges import FuturesExchangeProtocol

from .settings import reveal_secrets

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "robot_traders_lab.exchanges"
_DEMO_SETTING = "demo_trading"


async def load_exchange(
    exchange_config: dict[str, Any], **auto_params: Any
) -> FuturesExchangeProtocol:
    """Factory that loads the exchange a config names.

    The entry point callable receives the config and returns an exchange instance.
    It can be a class (called via __init__), a factory function, or any callable.
    If the callable returns a coroutine, it will be awaited.

    Args:
        exchange_config: Must name the exchange to load, as one of:
            - A short name registered via entry points (e.g., "bitget", "simulator")
            - A full module path (e.g., "my_exchange.factory:create_exchange")
    """
    exchange_config.update(**auto_params)
    exchange_name = _extract_exchange_name(exchange_config)
    factory = _load_exchange_factory(exchange_name)
    declared = _declared_settings(factory)
    _refuse_a_demo_the_connector_does_not_offer(
        exchange_name, declared, exchange_config
    )
    _warn_about_settings_the_connector_does_not_take(
        exchange_name, declared, exchange_config
    )
    exchange = await _create_exchange(factory, reveal_secrets(exchange_config))
    logger.debug(f"Exchange `{exchange_name}` loaded")
    return exchange


def _extract_exchange_name(config: dict[str, Any]) -> str:
    try:
        name: str = config.pop("exchange")
        return name
    except KeyError as e:
        raise ExchangeCriticalError("`exchange` is a required config key") from e


def _load_exchange_factory(exchange_name: str) -> Any:
    try:
        return load_class(exchange_name, _ENTRY_POINT_GROUP)
    except ClassLoadingError as e:
        raise ExchangeCriticalError(str(e)) from e


def _declared_settings(factory: Any) -> Mapping[str, inspect.Parameter]:
    return inspect.signature(factory).parameters


def _refuse_a_demo_the_connector_does_not_offer(
    exchange_name: str,
    declared: Mapping[str, inspect.Parameter],
    settings: dict[str, Any],
) -> None:
    """A connector that does not name `demo_trading` among its settings trades
    real money whatever the account asks.
    """
    if not settings.get(_DEMO_SETTING) or _DEMO_SETTING in declared:
        return
    raise ExchangeCriticalError(
        f"Exchange connector '{exchange_name}' offers no demo environment; "
        f"the account asks for one with `{_DEMO_SETTING}`, so it does not trade"
    )


def _warn_about_settings_the_connector_does_not_take(
    exchange_name: str,
    declared: Mapping[str, inspect.Parameter],
    settings: dict[str, Any],
) -> None:
    """Say which configured settings this connector has no use for.

    The exchange is still built, on its own defaults, because a setting it
    cannot use is not a reason to stop trading.
    """
    unknown = settings_the_factory_does_not_take(declared, settings)
    if unknown:
        logger.warning(
            "Exchange connector '%s' takes no %s; it will use its own defaults",
            exchange_name,
            ", ".join(f"`{setting}`" for setting in unknown),
        )


async def _create_exchange(
    factory: Any, config: dict[str, Any]
) -> FuturesExchangeProtocol:
    result = factory(**config)
    if inspect.iscoroutine(result):
        return cast(FuturesExchangeProtocol, await result)
    return cast(FuturesExchangeProtocol, result)
