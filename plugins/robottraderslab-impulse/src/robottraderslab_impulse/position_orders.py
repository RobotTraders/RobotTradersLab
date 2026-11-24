from robottraderslab.strategies import (
    AccountSnapshot,
    BookKeeper,
    PositionSide,
    TrackedPosition,
)
from robottraderslab.strategies.futures import FuturesAccount

from .position_protection import stop_loss_trigger_price
from .profile_config import ProfileConfig
from .profile_snapshot import ProfileSnapshot

_ENTRY_REASONS = {
    PositionSide.LONG: "impulse long entry",
    PositionSide.SHORT: "impulse short entry",
}
_EXIT_REASONS = {
    PositionSide.LONG: "impulse long exit",
    PositionSide.SHORT: "impulse short exit",
}


def wanted_entry_side(
    profile: ProfileConfig, profile_snapshot: ProfileSnapshot
) -> PositionSide | None:
    """Return the side this profile wants to open on this candle, or None.

    A profile decides alone. Profiles stacked on one symbol may disagree, and
    the venue nets their orders into the single position it holds.
    """
    if not profile.short_only and profile_snapshot.long_entry:
        return PositionSide.LONG
    if not profile.long_only and profile_snapshot.short_entry:
        return PositionSide.SHORT
    return None


def wants_exit(position: TrackedPosition, profile_snapshot: ProfileSnapshot) -> bool:
    """Report whether the held position's own side is being told to leave."""
    if position.side == PositionSide.LONG:
        return profile_snapshot.long_exit
    return profile_snapshot.short_exit


def book_entry(
    account: FuturesAccount,
    profile: ProfileConfig,
    side: PositionSide,
    profile_snapshot: ProfileSnapshot,
    account_snapshot: AccountSnapshot,
    bookkeeper: BookKeeper,
) -> None:
    symbol = profile.symbol
    entry = (
        account.long_entry(symbol)
        if side == PositionSide.LONG
        else account.short_entry(symbol)
    )
    entry = (
        entry.size(profile.sizing, profile_snapshot.price, account_snapshot)
        .tag(profile.order_tag)
        .reason(_ENTRY_REASONS[side])
    )
    if profile.stop_loss_pct is not None:
        entry = entry.stop_loss(
            price=stop_loss_trigger_price(
                side, profile_snapshot.price, profile.stop_loss_pct
            )
        )
    bookkeeper.add(entry.build())


def book_exit(
    account: FuturesAccount,
    profile: ProfileConfig,
    position: TrackedPosition,
    bookkeeper: BookKeeper,
) -> None:
    exit_order = (
        account.close_tracked_position(profile.symbol, position)
        .tag(profile.order_tag)
        .reason(_EXIT_REASONS[position.side])
    )
    bookkeeper.add(exit_order.build())
