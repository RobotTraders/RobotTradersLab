from dataclasses import dataclass, field
from typing import ClassVar

from ...order import PlacedOrder, ProtectionKind, ProtectionVenue, TriggerType
from .base_action import BaseExchangeAction


@dataclass(kw_only=True, slots=True)
class PositionProtectionAction(BaseExchangeAction):
    """Once the action ran, `held` carries the order the venue acknowledged for
    the protection, under the id it reports among its open orders, and stays
    empty when the venue holds none. `position_gone` says the venue reported
    no position on the symbol, so nothing on it is left to protect.
    """

    protection: ClassVar[ProtectionKind]

    held: tuple[PlacedOrder, ...] = field(default=(), init=False)
    position_gone: bool = field(default=False, init=False)

    @property
    def level(self) -> str:
        """Where the protection is asked to stand, as a log line names it."""
        raise NotImplementedError

    @property
    def venue(self) -> ProtectionVenue:
        raise NotImplementedError


@dataclass(kw_only=True, slots=True)
class LevelProtectionAction(PositionProtectionAction):
    """A protection that stands at a price the strategy booked."""

    trigger_price: float
    trigger_type: TriggerType = TriggerType.DEFAULT

    @property
    def level(self) -> str:
        return str(self.trigger_price)
