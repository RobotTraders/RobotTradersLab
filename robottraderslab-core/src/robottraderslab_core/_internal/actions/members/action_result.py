from dataclasses import dataclass

from ...order import OnFillRead, OrderPlacement, PlacedOrder


@dataclass(kw_only=True, frozen=True)
class OrderOutcome:
    """One order an action placed, with the callback that follows it up."""

    placed_order: PlacedOrder
    placement: OrderPlacement
    on_filled: OnFillRead | None = None


@dataclass(kw_only=True, frozen=True)
class ActionResult:
    """Result of executing a trading action."""

    orders: tuple[OrderOutcome, ...] = ()
