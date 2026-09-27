from dataclasses import dataclass

from robottraderslab._core import ActionResult, BaseExchangeAction, Symbol

from .futures_exchange_protocol import FuturesExchangeProtocol


@dataclass(kw_only=True, slots=True)
class CancelOrderByIdAction(BaseExchangeAction):
    """Cancels one open order, named by the id the venue assigned it.

    Attributes:
        order_id: Exchange-assigned order id
    """

    exchange: FuturesExchangeProtocol
    symbol: Symbol
    order_id: str

    async def execute(self) -> ActionResult:
        await self.exchange.cancel_order_by_id(self.symbol, self.order_id)
        return ActionResult()
