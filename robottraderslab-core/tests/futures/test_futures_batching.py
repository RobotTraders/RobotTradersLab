import asyncio
import logging
from unittest.mock import AsyncMock, Mock

import pytest

from robottraderslab_core import Symbol
from robottraderslab_core.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ExchangeTransientError,
    InsufficientMarginError,
)
from robottraderslab_core.exchanges import (
    FuturesExchangeBase,
    FuturesExchangeProtocol,
    OrderModifyRequest,
    OrderProtocol,
    OrderRequest,
    OrderSide,
    PlacedOrder,
)
from robottraderslab_core.futures import place_in_order
from robottraderslab_core.futures.futures_batching import (
    modify_overlapped,
    place_overlapped,
)

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


class TestARefusedOrderOfABatch:
    async def test_an_exit_is_an_error_naming_the_profile(self, caplog):
        venue = self._venue_refusing(
            ExchangeRecoverableError("25202 Insufficient balance")
        )

        with caplog.at_level(logging.INFO):
            placed = await self._place(venue, is_exit=True)

        assert placed == [None]
        assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
            (
                logging.ERROR,
                "Exit refused, the strategy's position is not closed: BTC/USDT:USDT, "
                "profile BTC/USDT:USDT@4h-fast, short market entry, part of an exit: "
                "25202 Insufficient balance",
            )
        ]

    async def test_an_exit_refused_when_a_resting_order_is_replaced_is_an_error(
        self, caplog
    ):
        resting = Mock(spec=OrderProtocol)
        resting.order_id = "o1"
        resting.reduce_only = None
        venue = self._venue_refusing(ExchangeRecoverableError("refused"))
        venue.get_open_orders = AsyncMock(  # type: ignore[method-assign]
            spec=FuturesExchangeProtocol.get_open_orders, return_value=[resting]
        )
        venue.cancel_order_by_id = AsyncMock(  # type: ignore[method-assign]
            spec=FuturesExchangeProtocol.cancel_order_by_id
        )
        replacement = OrderModifyRequest(
            order_id="o1", order=self._order_request(is_exit=True)
        )

        with caplog.at_level(logging.INFO):
            placed = await modify_overlapped(venue, [replacement])

        assert placed == [None]
        assert [record.levelno for record in caplog.records] == [logging.ERROR]

    async def test_an_entry_refused_for_margin_is_a_warning(self, caplog):
        venue = self._venue_refusing(
            InsufficientMarginError("25202 Insufficient balance")
        )

        with caplog.at_level(logging.INFO):
            await self._place(venue, is_exit=False)

        assert [record.levelno for record in caplog.records] == [logging.WARNING]

    async def test_a_transient_answer_is_raised_and_never_reported(self, caplog):
        venue = self._venue_refusing(ExchangeTransientError("timeout"))

        with caplog.at_level(logging.INFO), pytest.raises(ExchangeTransientError):
            await self._place(venue, is_exit=True)

        assert caplog.records == []

    def _order_request(self, *, is_exit: bool) -> OrderRequest:
        return OrderRequest(
            symbol=BTC_USDT,
            side=OrderSide.SELL,
            quantity=1.0,
            client_order_id="bot0launch-3-4h-fast",
            is_exit=is_exit,
        )

    async def _place(
        self, venue: FuturesExchangeBase, *, is_exit: bool
    ) -> list[PlacedOrder | None]:
        return await place_overlapped(venue, [self._order_request(is_exit=is_exit)])

    def _venue_refusing(self, error: ExchangeRecoverableError) -> FuturesExchangeBase:
        venue = FuturesExchangeBase.__new__(FuturesExchangeBase)
        venue.place_market_order = AsyncMock(  # type: ignore[method-assign]
            spec=FuturesExchangeProtocol.place_market_order, side_effect=error
        )
        return venue
