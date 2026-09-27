import math
from dataclasses import dataclass

from robottraderslab._core import ActionResult, BaseExchangeAction, Symbol

from .futures_exchange_protocol import FuturesExchangeProtocol


@dataclass(kw_only=True, slots=True)
class SetLeverageAction(BaseExchangeAction):
    """Sets the leverage the account trades the symbol at."""

    exchange: FuturesExchangeProtocol
    symbol: Symbol
    leverage: float

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: If the leverage is below 1 or NaN.
        """
        if self.leverage < 1 or math.isnan(self.leverage):
            raise ValueError(
                f"`leverage` must be at least 1; received {self.leverage} on "
                f"{self.symbol}"
            )

    async def execute(self) -> ActionResult:
        await self.exchange.set_leverage(self.symbol, self.leverage)
        return ActionResult()
