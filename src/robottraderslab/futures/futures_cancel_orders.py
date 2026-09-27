from dataclasses import dataclass

from robottraderslab._core import ActionResult, BaseExchangeAction, Symbol, tag_of

from .futures_batching import cancel_overlapped
from .futures_exchange_protocol import FuturesExchangeProtocol


@dataclass(kw_only=True, slots=True)
class CancelOrdersForSymbolAction(BaseExchangeAction):
    """Cancels the open orders on the symbol, protections included, or only
    the ones carrying `order_tag` when it is set.
    """

    exchange: FuturesExchangeProtocol
    symbol: Symbol
    order_tag: str | None = None

    async def execute(self) -> ActionResult:
        if self.order_tag is None:
            await self.exchange.cancel_orders_for_symbol(self.symbol)
        else:
            await self._cancel_tagged_orders(self.order_tag)
        return ActionResult()

    async def _cancel_tagged_orders(self, order_tag: str) -> None:
        orders = await self.exchange.get_open_orders([self.symbol])
        await cancel_overlapped(
            self.exchange,
            self.symbol,
            [
                order.order_id
                for order in orders
                if tag_of(order.client_order_id) == order_tag
            ],
        )
