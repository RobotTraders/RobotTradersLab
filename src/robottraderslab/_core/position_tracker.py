from dataclasses import dataclass
from typing import NewType, Protocol

from .position import PositionSide, PositionSnapshot

TrackingId = NewType("TrackingId", str)


@dataclass(slots=True, frozen=True)
class TrackedPosition:
    """The position one tracking id holds on its symbol.

    Attributes:
        quantity: What it holds, in the base currency, above zero.
    """

    side: PositionSide
    quantity: float

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: If the quantity is not positive.
        """
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got: {self.quantity}")


class PositionTracker(Protocol):
    """The position each declared tracking id holds, derived from the venue's
    own fills.

    A venue reports one position per symbol however many tracking ids share
    it, so each id's share is resolved from the tag its own orders carry, in a
    backtest and live alike; a fill the run never watched, a stop-loss firing
    between two candles, is attributed like any other. The record is brought
    in line with the venue on every candle before the strategy books.
    """

    def get(self, tracking_id: TrackingId) -> TrackedPosition | None:
        """Return the position held under the id, None when it holds none."""
        ...


def signed_position(position: TrackedPosition | PositionSnapshot) -> float:
    """`TrackedPosition` and `PositionSnapshot` share no base type, so this
    gives both one signed float to sum or compare directly.
    """
    if position.side == PositionSide.LONG:
        return position.quantity
    return -position.quantity
