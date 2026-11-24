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
class UpdatePositionTakeProfitAction(LevelProtectionAction):
    """Moves the take-profit on an open position to a new trigger price."""

    protection: ClassVar[ProtectionKind] = OrderType.TAKE_PROFIT

    tag: str = "tp"
    exchange: FuturesExchangeProtocol

    @property
    def venue(self) -> FuturesExchangeProtocol:
        return self.exchange

    async def execute(self) -> ActionResult:
        await hold_protection(
            self,
            partial(
                self.exchange.update_position_take_profit,
                self.symbol,
                self.trigger_price,
                self.trigger_type,
                self.client_order_id,
            ),
        )
        return ActionResult()
