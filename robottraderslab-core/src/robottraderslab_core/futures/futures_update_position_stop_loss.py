from dataclasses import dataclass
from functools import partial
from typing import ClassVar

from robottraderslab_core._internal import (
    ActionResult,
    LevelProtectionAction,
    OrderType,
    ProtectionKind,
)

from .futures_exchange_protocol import FuturesExchangeProtocol
from .futures_position_protection import hold_protection


@dataclass(kw_only=True, slots=True)
class UpdatePositionStopLossAction(LevelProtectionAction):
    """Moves the stop-loss guarding an open position to a new trigger price."""

    protection: ClassVar[ProtectionKind] = OrderType.STOP_LOSS

    tag: str = "sl"
    exchange: FuturesExchangeProtocol

    @property
    def venue(self) -> FuturesExchangeProtocol:
        return self.exchange

    async def execute(self) -> ActionResult:
        await hold_protection(
            self,
            partial(
                self.exchange.update_position_stop_loss,
                self.symbol,
                self.trigger_price,
                self.trigger_type,
                self.client_order_id,
            ),
        )
        return ActionResult()
