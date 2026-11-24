import asyncio
from unittest.mock import AsyncMock

import pytest

from robottraderslab_core import Symbol
from robottraderslab_core.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ExchangeTransientError,
)
from robottraderslab_core.exchanges import (
    FuturesExchangeBase,
    FuturesExchangeProtocol,
    OrderRequest,
    OrderSide,
    PlacedOrder,
)
from robottraderslab_core.futures import place_in_order
from robottraderslab_core.futures.futures_batching import place_overlapped

BTC_USDT = Symbol.create("BTC/USDT:USDT")


class TestBatchWaiting:
    """The two batch helpers differ only in whether they let the requests overlap."""

    class _TracingVenue(FuturesExchangeProtocol):
        def __init__(self) -> None:
            self.trace: list[str] = []

        async def place_market_order(self, symbol, side, quantity, **kwargs):
            self.trace.append(f"start {quantity}")
            await asyncio.sleep(0)
            self.trace.append(f"end {quantity}")
            return PlacedOrder(order_id=str(quantity))

    def _requests(self) -> list[OrderRequest]:
        return [
            OrderRequest(symbol=BTC_USDT, side=OrderSide.BUY, quantity=quantity)
            for quantity in (1.0, 2.0)
        ]

    async def test_overlapped_starts_both_before_either_finishes(self):
        venue = self._TracingVenue()

        await place_overlapped(venue, self._requests())

        assert venue.trace == ["start 1.0", "start 2.0", "end 1.0", "end 2.0"]

    async def test_in_order_finishes_each_before_the_next_starts(self):
        venue = self._TracingVenue()

        await place_in_order(venue, self._requests())

        assert venue.trace == ["start 1.0", "end 1.0", "start 2.0", "end 2.0"]

    async def test_one_by_one_each_order_carries_its_venue_params(self):
        carried_params: list[dict] = []

        async def place_limit_order(**order):
            carried_params.append(order["params"])
            return PlacedOrder(order_id="one")

        venue = self._TracingVenue()
        venue.place_limit_order = place_limit_order  # type: ignore[method-assign]
        request = OrderRequest(
            symbol=BTC_USDT,
            side=OrderSide.BUY,
            quantity=1.0,
            limit_price=50000.0,
            params={"bitget": {"venueOnly": "99"}},
        )

        await place_in_order(venue, [request])

        assert carried_params == [{"bitget": {"venueOnly": "99"}}]

    async def test_in_order_reports_a_rejected_order(self):
        venue = self._TracingVenue()
        venue.place_market_order = AsyncMock(  # type: ignore[method-assign]
            spec=FuturesExchangeBase.place_market_order,
            side_effect=[
                PlacedOrder(order_id="one"),
                ExchangeRecoverableError("margin"),
            ],
        )

        placed = await place_in_order(venue, self._requests())

        assert placed[0].order_id == "one"
        assert placed[1] is None

    async def test_in_order_transient_failure_fails_the_whole_batch(self):
        venue = self._TracingVenue()
        venue.place_market_order = AsyncMock(  # type: ignore[method-assign]
            spec=FuturesExchangeBase.place_market_order,
            side_effect=ExchangeTransientError("timeout"),
        )

        with pytest.raises(ExchangeTransientError):
            await place_in_order(venue, self._requests())

    async def test_a_critical_waits_for_the_orders_in_flight(self):
        answered: list[float] = []

        async def revoke_the_first(symbol, side, quantity, **kwargs):
            if quantity == 1.0:
                raise ExchangeCriticalError("key revoked")
            await asyncio.sleep(0.01)
            answered.append(quantity)
            return PlacedOrder(order_id=str(quantity))

        venue = self._TracingVenue()
        venue.place_market_order = revoke_the_first  # type: ignore[method-assign]

        with pytest.raises(ExchangeCriticalError, match="key revoked"):
            await place_overlapped(venue, self._requests())

        assert answered == [2.0]
