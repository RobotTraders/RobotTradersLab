from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from .order import OrderSide
from .symbol import Symbol


class PositionSide(StrEnum):
    """The direction of a position: long gains when price rises, short when it falls."""

    LONG = "long"
    SHORT = "short"

    @property
    def closing_side(self) -> OrderSide:
        return OrderSide.SELL if self == PositionSide.LONG else OrderSide.BUY


class PositionMode(StrEnum):
    """Position modes for futures trading"""

    ONE_WAY = "one_way"
    HEDGE = "hedge"


class PositionProtocol(Protocol):
    """Minimum fields a position exposes to the engine."""

    quantity: float
    side: PositionSide
    entry_time: datetime


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """An open position as the venue reported it when the snapshot was taken.

    Attributes:
        quantity: What the position holds, in the base currency; `side`
            carries the direction.
        average_entry_price: The price the position was opened at, averaged
            over its fills.
        leverage: The leverage the position is held at.
        liquidation_price: The price the venue closes the position at, None
            on a venue that reports none.
    """

    symbol: Symbol
    side: PositionSide
    quantity: float
    average_entry_price: float
    entry_time: datetime
    leverage: float
    liquidation_price: float | None
