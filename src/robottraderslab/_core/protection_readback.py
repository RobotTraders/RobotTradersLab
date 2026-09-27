import logging
from collections.abc import Iterable, Sequence

from .actions import execute_trading_actions
from .actions.members import BaseExchangeAction, PositionProtectionAction
from .exceptions import ExchangeRecoverableError
from .order import OrderProtocol, ProtectionKind, ProtectionVenue
from .retry import BASE_DELAY_SECONDS, MAX_ATTEMPTS, retry_on_transient
from .symbol import Symbol

logger = logging.getLogger(__name__)


async def confirm_booked_protections(
    actions: Iterable[BaseExchangeAction],
    *,
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY_SECONDS,
) -> None:
    """A venue that acknowledges an update and does not hold it leaves the
    position at a level the strategy did not book until the next cycle, so
    every protection the venue acknowledged is read back before this one ends.
    The read starts once every booked action has run, so no order waits on it.
    A position the cycle closed has nothing left to protect, so its symbol is
    left out.

    Args:
        actions: The actions the cycle executed.
    """
    for protections in _latest_held_by_venue(actions).values():
        venue = protections[0].venue
        try:
            await _confirm_on_venue(venue, protections, max_attempts, base_delay)
        except ExchangeRecoverableError as error:
            logger.warning(
                "Protections on %s could not be read back: %s",
                _symbols_of(protections),
                error,
            )


def _latest_held_by_venue(
    actions: Iterable[BaseExchangeAction],
) -> dict[int, list[PositionProtectionAction]]:
    """A venue drops a position's protections of every kind with the position
    they guarded.
    """
    latest: dict[tuple[int, Symbol, ProtectionKind], PositionProtectionAction] = {}
    closed: set[tuple[int, Symbol]] = set()
    for action in actions:
        if not isinstance(action, PositionProtectionAction):
            continue
        if action.position_gone:
            closed.add((id(action.venue), action.symbol))
        if action.held:
            latest[(id(action.venue), action.symbol, action.protection)] = action
    by_venue: dict[int, list[PositionProtectionAction]] = {}
    for (venue_id, symbol, _), action in latest.items():
        if (venue_id, symbol) not in closed:
            by_venue.setdefault(venue_id, []).append(action)
    return by_venue


async def _confirm_on_venue(
    venue: ProtectionVenue,
    protections: Sequence[PositionProtectionAction],
    max_attempts: int,
    base_delay: float,
) -> None:
    missing = await _not_resting(venue, protections, max_attempts, base_delay)
    if not missing:
        return
    for action in missing:
        logger.warning(
            "%s: %s does not rest at %s, re-issuing it",
            action.symbol,
            action.protection,
            action.trigger_price,
        )
    await execute_trading_actions(
        missing, max_attempts=max_attempts, base_delay=base_delay
    )
    reissued = [action for action in missing if action.held]
    for action in await _not_resting(venue, reissued, max_attempts, base_delay):
        logger.error(
            "%s: %s does not hold at %s after being re-issued",
            action.symbol,
            action.protection,
            action.trigger_price,
        )


async def _not_resting(
    venue: ProtectionVenue,
    protections: Sequence[PositionProtectionAction],
    max_attempts: int,
    base_delay: float,
) -> list[PositionProtectionAction]:
    if not protections:
        return []
    orders: list[OrderProtocol] = await retry_on_transient(
        venue.get_open_orders,
        _symbols_of(protections),
        max_attempts=max_attempts,
        base_delay=base_delay,
    )
    resting_ids = {order.order_id for order in orders}
    return [
        action
        for action in protections
        if not any(placed.order_id in resting_ids for placed in action.held)
    ]


def _symbols_of(protections: Sequence[PositionProtectionAction]) -> list[Symbol]:
    return list(dict.fromkeys(action.symbol for action in protections))
