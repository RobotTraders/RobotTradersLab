from collections.abc import Sequence
from dataclasses import dataclass

from robottraderslab import Symbol
from robottraderslab.strategies import (
    AccountSnapshot,
    BookKeeper,
    PositionSide,
    PositionTracker,
    TrackingId,
)
from robottraderslab.strategies.futures import FuturesAccount

from .profile_config import ProfileConfig


@dataclass(frozen=True, slots=True)
class _Protection:
    """The stop-loss one symbol's open profiles ask for, and who is asking."""

    stop_loss_pct: float
    profile_ids: frozenset[TrackingId]


def book_position_protections(
    account: FuturesAccount,
    profiles: Sequence[ProfileConfig],
    position_tracker: PositionTracker,
    account_snapshot: AccountSnapshot,
    closing_profile_ids: set[TrackingId],
    bookkeeper: BookKeeper,
) -> None:
    """Move each held position's stop-loss to its configured distance.

    A symbol whose every open profile is closing this candle keeps no position
    to protect, so its stop-loss is cancelled.
    """
    for symbol, protection in _protections(profiles, position_tracker).items():
        if protection.profile_ids <= closing_profile_ids:
            for order in account_snapshot.stop_loss_orders(symbol):
                bookkeeper.add(account.cancel_order(symbol, order.order_id))
            continue
        position = account_snapshot.position(symbol)
        if position is None:
            continue
        bookkeeper.add(
            account.move_stop_loss(
                symbol,
                stop_loss_trigger_price(
                    position.side,
                    position.average_entry_price,
                    protection.stop_loss_pct,
                ),
            )
        )


def stop_loss_trigger_price(
    side: PositionSide, reference_price: float, stop_loss_pct: float
) -> float:
    """Return the trigger price a stop-loss sits at, on either side."""
    if side == PositionSide.LONG:
        return reference_price * (1 - stop_loss_pct)
    return reference_price * (1 + stop_loss_pct)


def _protections(
    profiles: Sequence[ProfileConfig], position_tracker: PositionTracker
) -> dict[Symbol, _Protection]:
    """Collect the stop-loss each symbol holding a position asks for.

    Profiles sharing a symbol are validated to agree on the distance, so the
    first of them speaks for all.
    """
    open_profiles: dict[Symbol, list[ProfileConfig]] = {}
    for profile in profiles:
        if position_tracker.get(profile.profile_id) is not None:
            open_profiles.setdefault(profile.symbol, []).append(profile)
    return {
        symbol: _Protection(
            stop_loss_pct=stop_loss_pct,
            profile_ids=frozenset(profile.profile_id for profile in on_symbol),
        )
        for symbol, on_symbol in open_profiles.items()
        if (stop_loss_pct := on_symbol[0].stop_loss_pct) is not None
    }
