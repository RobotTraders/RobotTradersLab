from dataclasses import dataclass
from functools import partial
from typing import ClassVar

from robottraderslab_core._internal import (
    ActionResult,
    OrderProtocol,
    OrderType,
    PositionProtectionAction,
    ProtectionKind,
)

from .futures_exchange_protocol import FuturesExchangeProtocol
from .futures_position_protection import hold_protection


@dataclass(kw_only=True, slots=True)
class UpdatePositionTrailingStopAction(PositionProtectionAction):
    """Has the venue trail a stop behind the price of an open position.

    A trailing stop already resting on the symbol is kept: placing another
    would start its trail from the price of the moment and loosen the stop.

    Attributes:
        distance_ratio: The share of the price the stop stays behind the best
            price reached.
        resting: The trailing stops the account's snapshot showed on the
            symbol.
    """

    protection: ClassVar[ProtectionKind] = OrderType.TRAILING_STOP

    exchange: FuturesExchangeProtocol
    distance_ratio: float
    resting: tuple[OrderProtocol, ...]

    @property
    def level(self) -> str:
        return f"{self.distance_ratio:.2%} behind the price"

    @property
    def venue(self) -> FuturesExchangeProtocol:
        return self.exchange

    async def execute(self) -> ActionResult:
        """Place the trailing stop unless one rests.

        A rejection is sent once more and one that is refused again is logged
        at ERROR, and a symbol the venue holds no position on is dropped at
        INFO, as for a stop-loss.

        Raises:
            ExchangeTransientError: Propagated for the caller to retry.
        """
        if not self.resting:
            await hold_protection(
                self,
                partial(
                    self.exchange.update_position_trailing_stop_ratio,
                    self.symbol,
                    self.distance_ratio,
                ),
            )
        return ActionResult()
