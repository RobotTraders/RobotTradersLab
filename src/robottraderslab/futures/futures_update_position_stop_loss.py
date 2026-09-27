from dataclasses import dataclass
from typing import ClassVar

from robottraderslab._core import (
    ActionResult,
    OrderType,
    PositionProtectionAction,
    ProtectionKind,
)

from .futures_exchange_protocol import FuturesExchangeProtocol
from .futures_position_protection import hold_protection


@dataclass(kw_only=True, slots=True)
class UpdatePositionStopLossAction(PositionProtectionAction):
    """Moves the stop-loss guarding an open position to a new trigger price."""

    protection: ClassVar[ProtectionKind] = OrderType.STOP_LOSS

    exchange: FuturesExchangeProtocol

    @property
    def venue(self) -> FuturesExchangeProtocol:
        return self.exchange

    async def execute(self) -> ActionResult:
        await hold_protection(self, self.exchange.update_position_stop_loss)
        return ActionResult()
