from collections.abc import Callable, Iterable

from robottraderslab import Symbol
from robottraderslab.strategies import (
    BaseExchangeAction,
    BookKeeper,
    OrderProtocol,
    OrderSide,
    PositionSide,
)
from robottraderslab.strategies.futures import (
    BatchableOrderAction,
    FuturesAccount,
    OrderModification,
)

from .profile_config import ProfileConfig
from .profile_snapshot import ProfileSnapshot
from .rungs import WantedRung, rung_band_index

_ENTERED_SIDES = (
    (OrderSide.BUY, PositionSide.LONG),
    (OrderSide.SELL, PositionSide.SHORT),
)

type RungUpdater = Callable[
    [
        FuturesAccount,
        ProfileConfig,
        ProfileSnapshot,
        list[WantedRung],
        BookKeeper,
    ],
    None,
]


def book_cancel_and_place(
    account: FuturesAccount,
    profile: ProfileConfig,
    profile_snapshot: ProfileSnapshot,
    wanted: list[WantedRung],
    bookkeeper: BookKeeper,
) -> None:
    """Drop the resting rungs and place the wanted set once they are gone."""
    cancels = book_cancel_entries(account, profile, profile_snapshot, bookkeeper)

    orders: list[BatchableOrderAction] = [rung.order for rung in wanted]
    if orders:
        bookkeeper.add(account.place_orders(orders), after=cancels)


def book_modify_in_place(
    account: FuturesAccount,
    profile: ProfileConfig,
    profile_snapshot: ProfileSnapshot,
    wanted: list[WantedRung],
    bookkeeper: BookKeeper,
) -> None:
    """Move each resting rung to its own band, so the book is never empty."""
    symbol = profile.symbol
    holding = profile_snapshot.position is not None

    modifications: list[OrderModification] = []
    cancel_ids: list[str] = [] if holding else list(profile_snapshot.stray_order_ids)
    missing: list[BatchableOrderAction] = []
    for order_side, position_side in _ENTERED_SIDES:
        resting, unclaimed = _resting_by_band(
            profile, profile_snapshot.pending_rungs, order_side, position_side
        )
        cancel_ids.extend(unclaimed)
        for rung in wanted:
            if rung.order.side != order_side:
                continue
            claimed = resting.pop(rung.band_index, None)
            if claimed is None:
                missing.append(rung.order)
            else:
                modifications.append(
                    OrderModification(order_id=claimed.order_id, order=rung.order)
                )
        cancel_ids.extend(rung.order_id for rung in resting.values())

    if modifications:
        bookkeeper.add(account.modify_orders(modifications), after=())
    cancels = _cancel_each(account, symbol, cancel_ids, bookkeeper)
    if missing:
        bookkeeper.add(account.place_orders(missing), after=cancels)


def book_cancel_entries(
    account: FuturesAccount,
    profile: ProfileConfig,
    profile_snapshot: ProfileSnapshot,
    bookkeeper: BookKeeper,
) -> list[BaseExchangeAction]:
    """Clear the resting rungs without placing anything in their place."""
    if profile_snapshot.position is None:
        return [bookkeeper.add(account.cancel_orders(profile.symbol), after=())]
    return _cancel_each(
        account,
        profile.symbol,
        (rung.order_id for rung in profile_snapshot.pending_rungs),
        bookkeeper,
    )


def _resting_by_band(
    profile: ProfileConfig,
    pending_rungs: Iterable[OrderProtocol],
    order_side: OrderSide,
    position_side: PositionSide,
) -> tuple[dict[int, OrderProtocol], list[str]]:
    resting: dict[int, OrderProtocol] = {}
    unclaimed: list[str] = []
    for rung in pending_rungs:
        if rung.side != order_side:
            continue
        band_index = rung_band_index(profile, position_side, rung.client_order_id)
        if band_index is None or band_index in resting:
            unclaimed.append(rung.order_id)
        else:
            resting[band_index] = rung
    return resting, unclaimed


def _cancel_each(
    account: FuturesAccount,
    symbol: Symbol,
    order_ids: Iterable[str],
    bookkeeper: BookKeeper,
) -> list[BaseExchangeAction]:
    return [
        bookkeeper.add(account.cancel_order(symbol, order_id), after=())
        for order_id in order_ids
    ]
