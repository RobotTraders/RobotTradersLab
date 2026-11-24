from collections.abc import Iterable, Mapping
from dataclasses import replace
from decimal import Decimal
from operator import attrgetter

from .order import Execution, FillEffect, OrderFill, OrderSide
from .position import PositionSide, PositionSnapshot
from .symbol import Symbol


def attribute_fill_effects(
    executions: Iterable[Execution],
    positions: Mapping[Symbol, PositionSnapshot],
) -> list[Execution]:
    """Only a complete stream can be attributed: each execution is placed by
    where the symbol ended up, so one held back shifts every older one.

    Executions sharing an instant are walked in the reverse of the order the
    venue booked them, so the one that took the position to zero is the last
    of them.

    Args:
        executions: Every execution read for one account, whatever symbol.
        positions: The same account's open positions as of the end of that
            stream, which is the anchor every effect is measured back from.

    Returns:
        The executions in the order given.
    """
    stream = list(executions)
    derived: dict[str, FillEffect] = {}
    for symbol, symbol_stream in _by_symbol(stream).items():
        derived |= _walk_back(symbol_stream, _signed_quantity(positions.get(symbol)))
    return [
        replace(execution, effect=derived.get(execution.execution_id))
        for execution in stream
    ]


def settle_booked_fill(
    order_fill: OrderFill, executions: Iterable[Execution]
) -> OrderFill:
    """An order can fill in several executions, and a close among them wins
    whatever opened on the far side of it, since the close is what ended the
    position.

    Args:
        order_fill: A fill the executor reported, naming no effect.
        executions: The attributed executions of the fill's order.
    """
    chronological = sorted(executions, key=attrgetter("timestamp"))
    if not chronological:
        return order_fill
    effects = [e.effect for e in chronological if e.effect is not None]
    profits = [
        e.realised_profit for e in chronological if e.realised_profit is not None
    ]
    return replace(
        order_fill,
        effect=_order_effect(effects),
        realised_profit=sum(profits) if profits else None,
    )


def _by_symbol(stream: list[Execution]) -> dict[Symbol, list[Execution]]:
    by_symbol: dict[Symbol, list[Execution]] = {}
    for execution in stream:
        by_symbol.setdefault(execution.symbol, []).append(execution)
    return by_symbol


def _signed_quantity(position: PositionSnapshot | None) -> Decimal:
    if position is None:
        return Decimal(0)
    quantity = _exact(position.quantity)
    return -quantity if position.side == PositionSide.SHORT else quantity


def _walk_back(stream: list[Execution], ended_on: Decimal) -> dict[str, FillEffect]:
    effects: dict[str, FillEffect] = {}
    after = ended_on
    for execution in reversed(sorted(stream, key=lambda one: one.timestamp)):
        before = after - _signed_delta(execution)
        effect = _effect_between(before, after)
        if effect is not None:
            effects[execution.execution_id] = effect
        after = before
    return effects


def _signed_delta(execution: Execution) -> Decimal:
    quantity = _exact(execution.quantity)
    return quantity if execution.side == OrderSide.BUY else -quantity


def _effect_between(before: Decimal, after: Decimal) -> FillEffect | None:
    """A position crossing zero in one move has closed, whatever it opened on the way."""
    if before == after:
        return None
    if not before:
        return "open"
    if not after or (before > 0) != (after > 0):
        return "close"
    return "increase" if abs(after) > abs(before) else "reduce"


def _exact(quantity: float) -> Decimal:
    """A walk arrives at an exact zero on a position that closed, holding no residue."""
    return Decimal(str(quantity))


def _order_effect(effects: list[FillEffect]) -> FillEffect | None:
    if "close" in effects:
        return "close"
    return effects[0] if effects else None
