from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from copy import copy
from dataclasses import replace
from typing import Any

from robottraderslab_core.engine import ActionID, StepCount
from robottraderslab_core.exchanges import BaseExchangeAction
from robottraderslab_core.futures import (
    BatchableOrderAction,
    ClosePositionAction,
    FuturesLimitOrderAction,
    FuturesMarketOrderAction,
    FuturesOrderBatchAction,
    FuturesOrderModifyAction,
)


def rebuilt_actions(
    actions: Iterable[BaseExchangeAction],
    gone: Collection[ActionID],
    replaced: Mapping[ActionID, BatchableOrderAction],
) -> list[BaseExchangeAction]:
    """Return the actions as they go out.

    The orders `gone` are left out, and the orders in `replaced` stand in for
    those they replace.

    Args:
        actions: Everything booked this candle, in booking order.
        gone: Ids of the orders to leave out, a member of a batch or of a
            modification among them.
        replaced: The order to send in place of each order, by the id of the
            order it replaces.

    Returns:
        The actions in booking order. An action nothing changes is itself, one
        whose members are left out or replaced is a copy carrying the members
        left, and one with no member left is absent.
    """
    return [
        rebuilt
        for action in actions
        if (rebuilt := _rebuilt_action(action, gone, replaced)) is not None
    ]


def carried_orders(
    carrier: FuturesOrderBatchAction | FuturesOrderModifyAction,
) -> Iterator[BatchableOrderAction]:
    """Yield the orders a batch places or a modification replaces with."""
    if isinstance(carrier, FuturesOrderBatchAction):
        yield from carrier.orders
    else:
        for replacement in carrier.modifications:
            yield replacement.order


def carried_items(
    carrier: FuturesOrderBatchAction | FuturesOrderModifyAction,
) -> Iterator[tuple[BaseExchangeAction, BatchableOrderAction]]:
    """Yield each order the batch or the modification carries, with its action."""
    for order in carried_orders(carrier):
        yield carrier, order


def booked_items(
    actions: Iterable[BaseExchangeAction],
) -> Iterator[tuple[BaseExchangeAction, BatchableOrderAction | ClosePositionAction]]:
    """Yield each order and each close the actions place, with its action.

    The action is the order's own, or the batch the order is a member of. The
    replacement a modification carries places nothing and is left out.
    """
    for action in actions:
        if isinstance(action, FuturesOrderBatchAction):
            yield from carried_items(action)
        elif isinstance(
            action,
            (ClosePositionAction, FuturesLimitOrderAction, FuturesMarketOrderAction),
        ):
            yield action, action


def closed_steps(steps: StepCount, closing_ratio: float) -> StepCount:
    """Return how many of the steps a close of that share takes off."""
    return round(steps * closing_ratio)


def is_immediate(order: BatchableOrderAction) -> bool:
    """Whether the order is sent at once: a market order waiting on no trigger."""
    return isinstance(order, FuturesMarketOrderAction) and order.trigger_price is None


def is_sent_at_once(booked: BatchableOrderAction | ClosePositionAction) -> bool:
    """Whether the order, or the close, is sent at once and does not rest."""
    if isinstance(booked, ClosePositionAction):
        return booked.limit_price is None
    return is_immediate(booked)


def altered[T](item: T, **fields: Any) -> T:
    """Return a copy of the item with the fields set, keeping the rest and its id."""
    copied = copy(item)
    for name, value in fields.items():
        setattr(copied, name, value)
    return copied


def without_protections[T: BatchableOrderAction](order: T) -> T:
    """Return a copy of the order carrying no stop-loss and no take-profit."""
    return altered(order, stop_loss=None, take_profit=None)


def _rebuilt_action(
    action: BaseExchangeAction,
    gone: Collection[ActionID],
    replaced: Mapping[ActionID, BatchableOrderAction],
) -> BaseExchangeAction | None:
    if action.id in gone:
        return None
    if isinstance(action, (FuturesLimitOrderAction, FuturesMarketOrderAction)):
        return _standing_in_for(action, replaced)
    if isinstance(action, FuturesOrderBatchAction):
        return _rebuilt_batch(action, gone, replaced)
    if isinstance(action, FuturesOrderModifyAction):
        return _rebuilt_modification(action, gone, replaced)
    return action


def _standing_in_for[T: BatchableOrderAction](
    order: T, replaced: Mapping[ActionID, BatchableOrderAction]
) -> T | BatchableOrderAction:
    return replaced.get(order.id, order)


def _rebuilt_batch(
    batch: FuturesOrderBatchAction,
    gone: Collection[ActionID],
    replaced: Mapping[ActionID, BatchableOrderAction],
) -> FuturesOrderBatchAction | None:
    return _rebuilt_carrier(
        batch,
        "orders",
        batch.orders,
        gone,
        replaced,
        lambda order: order,
        lambda _, order: order,
    )


def _rebuilt_modification(
    action: FuturesOrderModifyAction,
    gone: Collection[ActionID],
    replaced: Mapping[ActionID, BatchableOrderAction],
) -> FuturesOrderModifyAction | None:
    return _rebuilt_carrier(
        action,
        "modifications",
        action.modifications,
        gone,
        replaced,
        lambda replacement: replacement.order,
        lambda replacement, order: replace(replacement, order=order),
    )


def _rebuilt_carrier[A: (FuturesOrderBatchAction, FuturesOrderModifyAction), M](
    action: A,
    field: str,
    members: Sequence[M],
    gone: Collection[ActionID],
    replaced: Mapping[ActionID, BatchableOrderAction],
    order_of: Callable[[M], BatchableOrderAction],
    with_order: Callable[[M, BatchableOrderAction], M],
) -> A | None:
    rebuilt_members = _rebuilt_members(members, gone, replaced, order_of, with_order)
    if rebuilt_members is None:
        return action
    if not rebuilt_members:
        return None
    return altered(
        action, **{field: rebuilt_members}, symbol=order_of(rebuilt_members[0]).symbol
    )


def _rebuilt_members[M](
    members: Sequence[M],
    gone: Collection[ActionID],
    replaced: Mapping[ActionID, BatchableOrderAction],
    order_of: Callable[[M], BatchableOrderAction],
    with_order: Callable[[M, BatchableOrderAction], M],
) -> list[M] | None:
    if not any(
        order_of(member).id in gone or order_of(member).id in replaced
        for member in members
    ):
        return None
    return [
        with_order(member, _standing_in_for(order_of(member), replaced))
        for member in members
        if order_of(member).id not in gone
    ]
