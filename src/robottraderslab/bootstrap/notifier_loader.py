import inspect
import logging
from enum import StrEnum
from typing import Any

from robottraderslab._core import (
    OnOrderFilled,
    OnOrderPlaced,
    PlacementNotifier,
    load_class,
)
from robottraderslab.exceptions import ExchangeCriticalError, StrategyCriticalError

from .settings import SecretsByName, resolve_secret_references, reveal_secrets

logger = logging.getLogger(__name__)

_NOTIFIER_ENTRY_POINT_GROUP = "robot_traders_lab.notifiers"
_LOGGING_HANDLER_ENTRY_POINT_GROUP = "robot_traders_lab.logging_handlers"

type _NotifierConfigs = dict[str, dict[str, dict[str, Any]]]
type LogHandlers = dict[str, dict[str, Any]]
type UnavailableNotifiers = dict[str, str]


class _Occasion(StrEnum):
    """The moments a notifier plugin can subscribe to from configuration."""

    FILLS = "fills"
    LOG = "log"
    PLACEMENTS = "placements"


def load_notifiers(
    notifier_configs: _NotifierConfigs,
    secrets_by_name: SecretsByName,
) -> tuple[list[OnOrderFilled], list[OnOrderPlaced]]:
    """Load the fill and placement subscribers a config declares.

    A notifier reports on a cycle without ever taking part in it, so one that
    cannot be built is left out of it.

    Args:
        notifier_configs: Plugin name to its occasion tables; each table is
            forwarded as kwargs to that plugin's factory.
        secrets_by_name: Secrets keyed by name, used to resolve a
            ``secret_name`` reference such as a Discord ``webhook_url``.

    Returns:
        The fill subscribers and the placement subscribers a cycle reports
        itself to.
    """
    on_fill: list[OnOrderFilled] = []
    on_placement: list[OnOrderPlaced] = []
    for plugin_name, occasions in notifier_configs.items():
        try:
            _subscribe(plugin_name, occasions, secrets_by_name, on_fill, on_placement)
        except (Exception, ExchangeCriticalError, StrategyCriticalError):
            logger.error(
                "Running without the `%s` notifier", plugin_name, exc_info=True
            )
    return on_fill, on_placement


def _subscribe(
    plugin_name: str,
    occasions: dict[str, dict[str, Any]],
    secrets: SecretsByName,
    on_fill: list[OnOrderFilled],
    on_placement: list[OnOrderPlaced],
) -> None:
    for occasion_name, config in resolve_secret_references(occasions, secrets).items():
        occasion = _validated_occasion(plugin_name, occasion_name)
        if occasion is _Occasion.LOG:
            continue
        notifier = load_class(plugin_name, _NOTIFIER_ENTRY_POINT_GROUP)(
            **reveal_secrets(config)
        )
        if occasion is _Occasion.FILLS:
            on_fill.append(notifier)
        else:
            on_placement.append(_placement_subscriber(plugin_name, notifier))
        logger.debug("Loaded `%s` notifier for `%s`", plugin_name, occasion.value)


def load_log_handlers(
    notifier_configs: _NotifierConfigs,
    secrets_by_name: SecretsByName,
) -> tuple[LogHandlers, UnavailableNotifiers]:
    """Resolve the log occasion of a notifier config into dictConfig handlers.

    A notifier reports on a run without ever taking part in it, so one that
    cannot be built is left out of it.

    The handlers this returns are what configures logging, so nothing here can
    be logged yet; `report_unavailable_notifiers` states the absences once
    logging can carry them.

    Args:
        notifier_configs: The same plugin-to-occasion-tables mapping
            `load_notifiers` reads.
        secrets_by_name: Secrets keyed by name, resolving a ``secret_name``
            reference such as a Discord ``webhook_url``.

    Returns:
        The handlers, keyed by plugin name and ready for
        `logging.config.dictConfig` with `level`, `formatter` and `filters`
        travelling through untouched; and why each absent notifier is absent,
        keyed the same way.
    """
    handlers: LogHandlers = {}
    unavailable: UnavailableNotifiers = {}
    for plugin_name, occasions in notifier_configs.items():
        try:
            handler = _log_handler(plugin_name, occasions, secrets_by_name)
        except (Exception, ExchangeCriticalError, StrategyCriticalError) as e:
            unavailable[plugin_name] = str(e)
            continue
        if handler is not None:
            handlers[plugin_name] = handler
    return handlers, unavailable


def report_unavailable_notifiers(unavailable: UnavailableNotifiers) -> None:
    for plugin_name, failure in unavailable.items():
        logger.error("Running without the `%s` notifier: %s", plugin_name, failure)


def _log_handler(
    plugin_name: str, occasions: dict[str, dict[str, Any]], secrets: SecretsByName
) -> dict[str, Any] | None:
    for occasion_name, config in resolve_secret_references(occasions, secrets).items():
        if _validated_occasion(plugin_name, occasion_name) is not _Occasion.LOG:
            continue
        handler_factory = load_class(plugin_name, _LOGGING_HANDLER_ENTRY_POINT_GROUP)
        logger.debug("Loaded `%s` log handler", plugin_name)
        return {"()": handler_factory, **reveal_secrets(config)}
    return None


def _validated_occasion(plugin_name: str, occasion_name: str) -> _Occasion:
    try:
        return _Occasion(occasion_name)
    except ValueError as e:
        valid_occasions = ", ".join(occasion.value for occasion in _Occasion)
        raise RuntimeError(
            f"`{plugin_name}` declares unknown notifier occasion `{occasion_name}`; "
            f"expected one of: {valid_occasions}"
        ) from e


def _placement_subscriber(plugin_name: str, notifier: object) -> OnOrderPlaced:
    if not isinstance(notifier, PlacementNotifier) or not inspect.iscoroutinefunction(
        notifier.notify_placement
    ):
        raise RuntimeError(
            f"`{plugin_name}` is configured for `placements` "
            "but does not implement `notify_placement`"
        )
    return notifier.notify_placement
