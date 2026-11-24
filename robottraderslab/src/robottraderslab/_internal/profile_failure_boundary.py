import logging
from collections.abc import Callable
from dataclasses import replace
from enum import StrEnum
from typing import Any

from robottraderslab_core import Symbol
from robottraderslab_core.exceptions import (
    BotCriticalError,
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ProfileError,
)

from .dry_run_failure import DryRunFailure, failure_of
from .profile_protocol import ProfileProtocol
from .trading_system import TradingMode

logger = logging.getLogger(__name__)


class ProfileSlot(StrEnum):
    SIGNALS = "signals"
    BOOKING = "booking"


def profile_failure_boundary(
    trading_mode: TradingMode,
    slot: ProfileSlot,
    profile_id: str,
    run_slot: Callable[..., None],
    *arguments: Any,
) -> bool:
    """Run one profile's slot, so the failure of the profile stays its own.

    Live, an `Exception` the slot raises takes the profile out of the cycle and
    the other profiles trade. A `BotCriticalError` and an `ExchangeCriticalError`
    are never contained: the first stops the bot wherever it is raised, a slot
    included, except that a bot with a working copy runs the copy when the dry
    run of an edited file meets one, and the venue being unusable stops it for
    every profile. A `ProfileError` is logged without a traceback since the
    strategy stated it on purpose. Outside live the failure propagates, so a
    bug surfaces while the strategy is developed.

    Args:
        slot: The slot the record names.
        profile_id: The profile the record names.
        run_slot: The slot's method of the profile's strategy.
        arguments: What `run_slot` is called with.

    Returns:
        False when a failure was contained, so work that depends on the slot
        having run through is left undone.
    """
    if trading_mode is not TradingMode.LIVE:
        run_slot(*arguments)
        return True
    try:
        run_slot(*arguments)
    except ExchangeRecoverableError as error:
        logger.warning(_FAILURE, profile_id, slot, type(error).__name__, error)
    except ProfileError as error:
        logger.error(_FAILURE, profile_id, slot, type(error).__name__, error)
    except Exception as error:
        logger.error(
            _FAILURE, profile_id, slot, type(error).__name__, error, exc_info=True
        )
    else:
        return True
    return False


def dry_run_boundary(
    failures: list[DryRunFailure],
    slot: ProfileSlot,
    profile: ProfileProtocol,
    run_slot: Callable[..., None],
    *arguments: Any,
) -> bool:
    """Run one profile's slot in a dry run, reporting its failure.

    The failure goes to the caller in place of the log. Every `Exception` and
    `BotCriticalError` the slot raises is contained: a dry run sends nothing,
    so there is no bot to stop. It is attributed to the profile's symbol. An
    `ExchangeCriticalError` is the venue's and belongs to no profile, so it
    propagates to the dry run's caller.

    Args:
        failures: Where the failure is appended.
        slot: The slot the message names.
        profile: The profile the message names and whose symbol the failure
            belongs to.
        run_slot: The slot's method of the profile's strategy.
        arguments: What `run_slot` is called with.

    Returns:
        False when a failure was reported, so work that depends on the slot
        having run through is left undone.
    """
    try:
        run_slot(*arguments)
    except ExchangeCriticalError:
        raise
    except (Exception, BotCriticalError) as error:
        symbol = getattr(profile, "symbol", None)
        message = _FAILURE % (profile.profile_id, slot, type(error).__name__, error)
        logger.debug(message, exc_info=True)
        failures.append(
            replace(
                failure_of(error, None if symbol is None else Symbol.create(symbol)),
                message=message,
            )
        )
        return False
    return True


_FAILURE = "Profile %s sat out this candle: the %s slot raised %s: %s"
