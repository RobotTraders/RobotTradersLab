from dataclasses import dataclass

from robottraderslab.strategies import AccountSnapshot, OrderSide, PositionSide, tag_of
from robottraderslab.strategies.futures import (
    BatchableOrderAction,
    FuturesAccount,
    PositionSnapshot,
)

from .entry_order import EntryOrder
from .position_protection import stop_loss_trigger_price
from .profile_config import ProfileConfig
from .profile_snapshot import ProfileSnapshot
from .reentry_guard import ReentryState


@dataclass(frozen=True, slots=True)
class WantedRung:
    """An entry rung the strategy wants resting at one of its bands."""

    band_index: int
    order: BatchableOrderAction


def entered_side(order_side: OrderSide) -> PositionSide:
    return PositionSide.LONG if order_side is OrderSide.BUY else PositionSide.SHORT


def rung_band_index(
    profile: ProfileConfig, side: PositionSide, client_order_id: str | None
) -> int | None:
    """A rung's tag names its side."""
    tag = tag_of(client_order_id)
    return None if tag is None else profile.band_index_by_tag[side].get(tag)


def is_past_stop_loss(
    side: PositionSide, band_price: float, last_close: float, stop_loss_pct: float
) -> bool:
    """Whether the last close has reached the stop-loss of a rung on this band.

    An entry placed there would be stopped out on the candle it fills on.
    """
    stop_loss = stop_loss_trigger_price(side, band_price, stop_loss_pct)
    if side is PositionSide.LONG:
        return last_close <= stop_loss
    return last_close >= stop_loss


def wanted_while_flat(
    account: FuturesAccount,
    profile: ProfileConfig,
    profile_snapshot: ProfileSnapshot,
    reentry_state: ReentryState,
    account_snapshot: AccountSnapshot,
    entry_order: EntryOrder,
) -> list[WantedRung]:
    """Return a rung for every band the profile may enter on, both sides."""
    wanted: list[WantedRung] = []
    for side, band_prices in profile_snapshot.bands.items():
        if not reentry_state.is_entry_allowed(side):
            continue
        for band_index, band_price in enumerate(band_prices, start=1):
            if is_past_stop_loss(
                side, band_price, profile_snapshot.last_close, profile.stop_loss_pct
            ):
                continue
            wanted.append(
                _build_rung(
                    account,
                    profile,
                    side,
                    band_index,
                    band_price,
                    profile_snapshot.reference,
                    account_snapshot,
                    entry_order,
                )
            )
    return wanted


def wanted_while_holding(
    account: FuturesAccount,
    profile: ProfileConfig,
    profile_snapshot: ProfileSnapshot,
    position: PositionSnapshot,
    held_bands: frozenset[int],
    reentry_state: ReentryState,
    account_snapshot: AccountSnapshot,
    entry_order: EntryOrder,
) -> list[WantedRung]:
    """A band counts as filled only on the position tracker's record."""
    if not reentry_state.is_entry_allowed(position.side):
        return []

    wanted: list[WantedRung] = []
    for band_index, band_price in enumerate(
        profile_snapshot.bands[position.side], start=1
    ):
        if band_index in held_bands:
            continue
        if is_past_stop_loss(
            position.side,
            band_price,
            profile_snapshot.last_close,
            profile.stop_loss_pct,
        ):
            continue
        wanted.append(
            _build_rung(
                account,
                profile,
                position.side,
                band_index,
                band_price,
                profile_snapshot.reference,
                account_snapshot,
                entry_order,
            )
        )
    return wanted


def _build_rung(
    account: FuturesAccount,
    profile: ProfileConfig,
    side: PositionSide,
    band_index: int,
    band_price: float,
    reference: float,
    account_snapshot: AccountSnapshot,
    entry_order: EntryOrder,
) -> WantedRung:
    entry = (
        account.long_entry(profile.symbol)
        if side is PositionSide.LONG
        else account.short_entry(profile.symbol)
    )
    resting_at_band = entry_order.rest_at_band(entry, band_price)
    return WantedRung(
        band_index=band_index,
        order=resting_at_band.size(
            profile.rung_sizing[side], band_price, account_snapshot
        )
        .stop_loss(
            price=stop_loss_trigger_price(side, band_price, profile.stop_loss_pct)
        )
        .take_profit(price=reference)
        .tag(profile.rung_tags[side][band_index - 1])
        .build(),
    )
