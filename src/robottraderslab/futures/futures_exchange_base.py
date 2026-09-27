from collections.abc import Sequence

from robottraderslab._core import OrderModifyRequest, OrderRequest, PlacedOrder

from .futures_batching import modify_overlapped, place_overlapped
from .futures_exchange_protocol import FuturesExchangeProtocol


class FuturesExchangeBase(FuturesExchangeProtocol):
    """Batch fallbacks for venues without batch endpoints.

    Each batch travels as one request per order, overlapped. A venue with a
    batch endpoint overrides these so the batch travels as fewer requests.
    """

    async def place_orders(
        self, requests: Sequence[OrderRequest]
    ) -> list[PlacedOrder | None]:
        """Place one order per request, overlapping the requests."""
        return await place_overlapped(self, requests)

    async def modify_orders(
        self, requests: Sequence[OrderModifyRequest]
    ) -> list[PlacedOrder | None]:
        """Cancel and re-place one order per request, overlapping the requests."""
        return await modify_overlapped(self, requests)
