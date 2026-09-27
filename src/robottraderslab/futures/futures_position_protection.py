import logging
from collections.abc import Awaitable, Callable

from robottraderslab._core import PlacedOrder, PositionProtectionAction, Symbol
from robottraderslab.exceptions import (
    ExchangeRecoverableError,
    ExchangeTransientError,
    NoOpenPositionError,
)

logger = logging.getLogger(__name__)

_REJECTION_REPORTS = (
    (logging.WARNING, "%s: %s update to %s rejected, re-issuing it: %s"),
    (
        logging.ERROR,
        "%s: %s does not hold at %s, the venue rejected the update twice: %s",
    ),
)


async def hold_protection(
    action: PositionProtectionAction,
    update: Callable[[Symbol, float], Awaitable[PlacedOrder]],
) -> None:
    """A position guarded at a level the strategy did not book is the failure
    this prevents. A symbol the account holds no position on any more has
    nothing left to protect, so its update is dropped without alarm.

    Raises:
        ExchangeTransientError: Propagated for the caller to retry.
    """
    action.held = ()
    action.position_gone = False
    for level, message in _REJECTION_REPORTS:
        try:
            action.held = (await update(action.symbol, action.trigger_price),)
            return
        except NoOpenPositionError as gone:
            action.position_gone = True
            logger.info(
                "%s: %s update dropped, %s", action.symbol, action.protection, gone
            )
            return
        except ExchangeTransientError:
            raise
        except ExchangeRecoverableError as rejection:
            logger.log(
                level,
                message,
                action.symbol,
                action.protection,
                action.trigger_price,
                rejection,
            )
