import asyncio
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TypeGuard

from robottraderslab_core import Symbol
from robottraderslab_core.engine import (
    BASE_DELAY_SECONDS,
    MAX_ATTEMPTS,
    AccountProtocol,
    AccountRequirement,
    ActionID,
    StepCount,
    position_steps,
    to_quantity,
    to_step_count,
)
from robottraderslab_core.exceptions import ExchangeRecoverableError
from robottraderslab_core.exchanges import (
    BaseExchangeAction,
    OrderSide,
    PositionSnapshot,
)
from robottraderslab_core.futures import (
    BatchableOrderAction,
    ClosePositionAction,
    FuturesLimitOrderAction,
    FuturesOrderModifyAction,
)
from robottraderslab_core.strategies import (
    AccountSnapshots,
    BookKeeper,
    StrategyRequirements,
)

from .booked_orders import (
    altered,
    booked_items,
    carried_items,
    closed_steps,
    is_sent_at_once,
    rebuilt_actions,
    without_protections,
)

logger = logging.getLogger(__name__)

_DIRECTION: Mapping[OrderSide, int] = {OrderSide.BUY: 1, OrderSide.SELL: -1}


async def split_exits(
    requirements: StrategyRequirements,
    account_snapshots: AccountSnapshots,
    bookkeeper: BookKeeper,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY_SECONDS,
) -> None:
    """Split each exit into its reduce-only part and its plain remainder.

    The part that reduces the venue's position goes reduce-only and the rest as
    a plain order after it. An exit is judged against the position the account's
    snapshot holds, with the orders sent at once counted in booking order and a
    resting exit judged after all of them. An order that is not an exit counts
    only for what it adds to the position, since the venue may refuse it. An
    exit booked reduce-only, an exit that replaces a resting order, and every
    exit on a symbol whose tracked positions the engine cannot tie to the
    venue's, sends no plain remainder: what exceeds the position is cut and
    logged. The position of a symbol the snapshot holds none for, since the
    account declared no positions or not that symbol, is read from the venue,
    the accounts together, and only when an exit not booked reduce-only needs
    it. When that read fails, those exits go out reduce-only whole, which free
    margin never refuses and which opens no position; a venue that refuses a
    reduce-only order above its position refuses one that exceeds it, and a
    strategy that books its exit on every candle its side stays closed books
    it again the next cycle, and one that books it on the crossing candle alone
    does not.

    Args:
        requirements: What the strategy declared in `setup`.
        account_snapshots: The snapshots read for this candle.
        bookkeeper: Collector whose booked actions are replaced.
        max_attempts: Total retry attempts of a position read.
        base_delay: Base retry delay in seconds, doubled on each retry.

    Raises:
        ExchangeCriticalError: If the venue is unusable on a position read.
    """
    actions = bookkeeper.list_actions()
    if not _has_exit_to_split(_judged(actions)):
        return
    accounts = _accounts_with_exits(requirements, actions)
    unread = [
        _unread_symbols(requirement, items, account_snapshots)
        for requirement, items in accounts
    ]
    reads = await asyncio.gather(
        *(
            _read_positions(requirement.account, symbols, (max_attempts, base_delay))
            for (requirement, _), symbols in zip(accounts, unread, strict=True)
        )
    )
    changes = _Changes()
    unanchored = _unanchored_symbols(requirements)
    for (requirement, items), symbols, read in zip(
        accounts, unread, reads, strict=True
    ):
        _split_account(
            requirement, items, account_snapshots, symbols, read, unanchored, changes
        )
    if changes.is_empty:
        return
    rebuilt = rebuilt_actions(actions, changes.gone, changes.replaced)
    bookkeeper._replace_with_followers(
        _with_remainders(rebuilt, changes.remainders),
        _followers(changes.remainders),
    )


type _Booked = BatchableOrderAction | ClosePositionAction
type _Item = tuple[BaseExchangeAction, _Booked]


@dataclass(slots=True)
class _Changes:
    gone: set[ActionID] = field(default_factory=set)
    replaced: dict[ActionID, BatchableOrderAction] = field(default_factory=dict)
    remainders: dict[ActionID, list[BatchableOrderAction]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.gone or self.replaced or self.remainders)


class _Net:
    def __init__(self, steps: StepCount) -> None:
        self._steps = steps
        self._rooms: dict[OrderSide, StepCount] = {}

    def closable(self, side: OrderSide) -> StepCount:
        return max(0, -_DIRECTION[side] * self._steps)

    def close(self, closing_ratio: float) -> None:
        closed = closed_steps(abs(self._steps), closing_ratio)
        self._steps += -closed if self._steps > 0 else closed

    def grow(self, side: OrderSide, steps: StepCount) -> None:
        if self._steps * _DIRECTION[side] >= 0:
            self.move(side, steps)

    def move(self, side: OrderSide, steps: StepCount) -> None:
        self._steps += _DIRECTION[side] * steps

    def reserve_close(self, closing_ratio: float) -> None:
        side = OrderSide.SELL if self._steps > 0 else OrderSide.BUY
        self._rooms[side] -= closed_steps(self._rooms[side], closing_ratio)

    def room(self, side: OrderSide) -> StepCount:
        return self._rooms[side]

    def start_resting(self) -> None:
        self._rooms = {side: self.closable(side) for side in OrderSide}

    def take(self, side: OrderSide, steps: StepCount) -> None:
        self._rooms[side] -= steps


def _has_exit_to_split(items: Iterable[_Item]) -> bool:
    return any(_is_exit_to_split(booked) for _, booked in items)


def _is_exit_to_split(booked: _Booked) -> TypeGuard[BatchableOrderAction]:
    return booked.is_exit and not isinstance(booked, ClosePositionAction)


def _judged(actions: Sequence[BaseExchangeAction]) -> Iterator[_Item]:
    yield from booked_items(actions)
    for action in actions:
        if isinstance(action, FuturesOrderModifyAction):
            yield from carried_items(action)


def _accounts_with_exits(
    requirements: StrategyRequirements, actions: Sequence[BaseExchangeAction]
) -> list[tuple[AccountRequirement, list[_Item]]]:
    accounts = []
    for requirement in requirements.account._get_all():
        items = [
            item for item in _judged(actions) if requirement.account._books(item[0])
        ]
        if _has_exit_to_split(items):
            accounts.append((requirement, items))
    return accounts


def _unread_symbols(
    requirement: AccountRequirement,
    items: Iterable[_Item],
    account_snapshots: AccountSnapshots,
) -> list[Symbol]:
    declared = set(_declared_symbols(requirement, account_snapshots))
    margin_bound = {booked.symbol for _, booked in items if _is_margin_bound(booked)}
    return sorted(margin_bound - declared, key=str)


def _declared_symbols(
    requirement: AccountRequirement, account_snapshots: AccountSnapshots
) -> list[Symbol]:
    if account_snapshots._positions_of(requirement.account) is None:
        return []
    return list(dict.fromkeys(requirement.symbols))


def _is_margin_bound(booked: _Booked) -> TypeGuard[BatchableOrderAction]:
    return (
        _is_exit_to_split(booked) and not booked.reduce_only and booked.denial is None
    )


async def _read_positions(
    account: AccountProtocol, symbols: Sequence[Symbol], retries: tuple[int, float]
) -> dict[Symbol, PositionSnapshot] | None:
    if not symbols:
        return {}
    max_attempts, base_delay = retries
    try:
        return await account._read_positions(
            symbols, max_attempts=max_attempts, base_delay=base_delay
        )
    except ExchangeRecoverableError as error:
        logger.warning(
            "%s: the positions on %s could not be read, so its exits not booked "
            "reduce-only go out reduce-only whole: %s",
            account.name,
            ", ".join(map(str, symbols)),
            error,
        )
        return None


def _unanchored_symbols(requirements: StrategyRequirements) -> set[tuple[str, Symbol]]:
    unanchored: set[tuple[str, Symbol]] = set()
    for tracked, symbols in requirements.tracker._unanchored():
        unanchored.update((tracked.account.name, symbol) for symbol in symbols)
    return unanchored


def _split_account(
    requirement: AccountRequirement,
    items: Sequence[_Item],
    account_snapshots: AccountSnapshots,
    unread: Sequence[Symbol],
    read: Mapping[Symbol, PositionSnapshot] | None,
    unanchored: set[tuple[str, Symbol]],
    changes: _Changes,
) -> None:
    account = requirement.account
    declared = _declared_symbols(requirement, account_snapshots)
    if read is None:
        _send_reduce_only(items, unread, changes)
    known = {**(account_snapshots._positions_of(account) or {}), **(read or {})}
    for symbol in [*declared, *(unread if read is not None else [])]:
        _split_symbol(
            account,
            [item for item in items if item[1].symbol == symbol],
            _Net(position_steps(known.get(symbol))),
            (account.name, symbol) in unanchored,
            changes,
        )


def _send_reduce_only(
    items: Iterable[_Item], symbols: Sequence[Symbol], changes: _Changes
) -> None:
    for _, booked in items:
        if booked.symbol in symbols and _is_margin_bound(booked):
            changes.replaced[booked.id] = _reduce_only(booked)


def _reduce_only(order: BatchableOrderAction) -> BatchableOrderAction:
    return altered(order, reduce_only=True)


def _split_symbol(
    account: AccountProtocol,
    items: Iterable[_Item],
    net: _Net,
    unanchored: bool,
    changes: _Changes,
) -> None:
    resting: list[_Item] = []
    for item in items:
        if _rests(item):
            resting.append(item)
        else:
            _settle_immediate(account, item, net, unanchored, changes)
    net.start_resting()
    for item in resting:
        _settle_resting(account, item, net, unanchored, changes)


def _rests(item: _Item) -> bool:
    action, booked = item
    return isinstance(action, FuturesOrderModifyAction) or not is_sent_at_once(booked)


def _settle_immediate(
    account: AccountProtocol,
    item: _Item,
    net: _Net,
    unanchored: bool,
    changes: _Changes,
) -> None:
    action, booked = item
    if isinstance(booked, ClosePositionAction):
        net.close(booked.closing_ratio)
    elif booked.denial is not None:
        return
    elif booked.is_exit:
        reduced, remainder = _settle_exit(
            account, action, booked, net.closable(booked.side), unanchored, changes
        )
        net.move(booked.side, reduced + remainder)
    else:
        net.grow(booked.side, to_step_count(booked.quantity))


def _settle_exit(
    account: AccountProtocol,
    action: BaseExchangeAction,
    order: BatchableOrderAction,
    room: StepCount,
    unanchored: bool,
    changes: _Changes,
) -> tuple[StepCount, StepCount]:
    steps = to_step_count(order.quantity)
    reduced = min(steps, room)
    capped = (
        order.reduce_only or unanchored or isinstance(action, FuturesOrderModifyAction)
    )
    remainder = 0 if capped else steps - reduced
    _log_cut(order, steps - reduced - remainder)
    if reduced == 0:
        if capped:
            changes.gone.add(order.id)
        elif denial := _quantity_denial(account, order, order.quantity):
            changes.replaced[order.id] = altered(order, denial=denial)
        return 0, remainder
    if not order.reduce_only or reduced != steps:
        changes.replaced[order.id] = _reducing_part(order, reduced, steps)
    if remainder:
        changes.remainders.setdefault(action.id, []).append(
            _remainder(account, order, remainder)
        )
    return reduced, remainder


def _log_cut(order: BatchableOrderAction, cut: StepCount) -> None:
    if cut:
        logger.warning(
            "%s: the exit of %s is cut by %s, since only the rest of it "
            "reduces the position the venue holds",
            order.symbol,
            f"profile {order.tag}" if order.tag else "no profile",
            to_quantity(cut),
        )


def _quantity_denial(
    account: AccountProtocol, order: BatchableOrderAction, quantity: float
) -> str | None:
    price = order.price if isinstance(order, FuturesLimitOrderAction) else None
    return account._quantity_denial(
        order.symbol, quantity, price or order.trigger_price
    )


def _reducing_part(
    order: BatchableOrderAction, reduced: StepCount, steps: StepCount
) -> BatchableOrderAction:
    if reduced == steps:
        return _reduce_only(order)
    return altered(order, reduce_only=True, quantity=to_quantity(reduced))


def _remainder(
    account: AccountProtocol, order: BatchableOrderAction, steps: StepCount
) -> BatchableOrderAction:
    quantity = to_quantity(steps)
    return replace(
        without_protections(order),
        quantity=quantity,
        reduce_only=False,
        on_filled=None,
        denial=_quantity_denial(account, order, quantity),
    )


def _settle_resting(
    account: AccountProtocol,
    item: _Item,
    net: _Net,
    unanchored: bool,
    changes: _Changes,
) -> None:
    action, booked = item
    if isinstance(booked, ClosePositionAction):
        net.reserve_close(booked.closing_ratio)
    elif booked.denial is None and booked.is_exit:
        reduced, _ = _settle_exit(
            account, action, booked, net.room(booked.side), unanchored, changes
        )
        net.take(booked.side, reduced)


def _with_remainders(
    actions: Iterable[BaseExchangeAction],
    remainders: Mapping[ActionID, Sequence[BatchableOrderAction]],
) -> list[BaseExchangeAction]:
    sequence: list[BaseExchangeAction] = []
    for action in actions:
        sequence.append(action)
        sequence.extend(remainders.get(action.id, ()))
    return sequence


def _followers(
    remainders: Mapping[ActionID, Sequence[BatchableOrderAction]],
) -> dict[ActionID, ActionID]:
    followers: dict[ActionID, ActionID] = {}
    for leader, orders in remainders.items():
        followers.update((order.id, leader) for order in orders)
    return followers
