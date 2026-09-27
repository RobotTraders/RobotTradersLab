import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ExchangeTransientError,
    StrategyCriticalError,
)
from robottraderslab.exchanges import (
    FuturesExchangeBase,
    FuturesExchangeProtocol,
    OrderFill,
    OrderProtocol,
    OrderRequest,
    OrderSide,
    PlacedOrder,
    TimeInForce,
)
from robottraderslab.futures import FuturesAccount, OrderModification
from robottraderslab.futures.futures_batching import place_in_order, place_overlapped
from robottraderslab.futures.futures_limit_order import FuturesLimitOrderAction
from robottraderslab.futures.futures_market_order import FuturesMarketOrderAction

BTC_USDT = Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def limit_order() -> FuturesLimitOrderAction:
    return FuturesLimitOrderAction(
        exchange=Mock(spec=FuturesExchangeProtocol),
        symbol=BTC_USDT,
        side=OrderSide.BUY,
        quantity=1.0,
        price=50000.0,
        on_filled=AsyncMock(spec=_on_filled),
    )


@pytest.fixture
def market_order() -> FuturesMarketOrderAction:
    return FuturesMarketOrderAction(
        exchange=Mock(spec=FuturesExchangeProtocol),
        symbol=BTC_USDT,
        side=OrderSide.SELL,
        quantity=2.0,
    )


class TestBatchAction:
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        limit_order: FuturesLimitOrderAction,
        market_order: FuturesMarketOrderAction,
    ) -> None:
        self.exchange = Mock(spec=FuturesExchangeProtocol)
        self.account = FuturesAccount(self.exchange, name="test")
        self.limit_order = limit_order
        self.market_order = market_order
        self.batch = self.account.place_orders([limit_order, market_order])

    async def test_each_order_travels_with_its_time_in_force(self):
        self.exchange.place_orders = AsyncMock(
            spec=FuturesExchangeProtocol.place_orders,
            return_value=[PlacedOrder(order_id="1"), PlacedOrder(order_id="2")],
        )
        post_only = FuturesLimitOrderAction(
            exchange=self.exchange,
            symbol=BTC_USDT,
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            time_in_force=TimeInForce.POST_ONLY,
        )
        batch = self.account.place_orders([post_only, self.market_order])

        await batch.execute()

        (requests,) = self.exchange.place_orders.await_args.args
        assert [r.time_in_force for r in requests] == [
            TimeInForce.POST_ONLY,
            TimeInForce.GTC,
        ]

    async def test_orders_travel_as_one_request(self):
        self.exchange.place_orders = AsyncMock(
            spec=FuturesExchangeProtocol.place_orders,
            return_value=[PlacedOrder(order_id="1"), PlacedOrder(order_id="2")],
        )

        await self.batch.execute()

        (requests,) = self.exchange.place_orders.await_args.args
        assert [r.limit_price for r in requests] == [50000.0, None]
        assert [r.quantity for r in requests] == [1.0, 2.0]
        assert [r.client_order_id for r in requests] == [
            self.limit_order.client_order_id,
            self.market_order.client_order_id,
        ]

    async def test_each_order_keeps_its_own_follow_up(self):
        self.exchange.place_orders = AsyncMock(
            spec=FuturesExchangeProtocol.place_orders,
            return_value=[PlacedOrder(order_id="1"), PlacedOrder(order_id="2")],
        )

        batch_result = await self.batch.execute()

        first, second = batch_result.orders
        assert first.placed_order.order_id == "1"
        assert first.on_filled is self.limit_order.on_filled
        assert second.on_filled is None
        assert first.placement.kind == "limit"
        assert second.placement.kind == "market"

    async def test_each_placement_carries_its_orders_reason(self):
        reasoned = FuturesMarketOrderAction(
            exchange=self.exchange,
            symbol=BTC_USDT,
            side=OrderSide.SELL,
            quantity=2.0,
            reason="impulse long exit",
        )
        batch = self.account.place_orders([self.limit_order, reasoned])
        self.exchange.place_orders = AsyncMock(
            spec=FuturesExchangeProtocol.place_orders,
            return_value=[PlacedOrder(order_id="1"), PlacedOrder(order_id="2")],
        )

        batch_result = await batch.execute()

        first, second = batch_result.orders
        assert first.placement.reason is None
        assert second.placement.reason == "impulse long exit"

    async def test_order_the_venue_rejected(self):
        self.exchange.place_orders = AsyncMock(
            spec=FuturesExchangeProtocol.place_orders,
            return_value=[None, PlacedOrder(order_id="2")],
        )

        batch_result = await self.batch.execute()

        (outcome,) = batch_result.orders
        assert outcome.placed_order.order_id == "2"

    async def test_builders_are_placed_as_the_orders_they_build(self):
        batch = self.account.place_orders(
            [
                self.account.long_entry(BTC_USDT, 1.0).limit(50000.0),
                self.market_order,
            ]
        )

        assert [order.kind for order in batch.orders] == ["limit", "market"]
        assert batch.orders[0].price == 50000.0

    async def test_unsized_builder_in_a_batch(self):
        with pytest.raises(StrategyCriticalError, match="The order has no size"):
            self.account.place_orders([self.account.long_entry(BTC_USDT)])

    async def test_batch_without_orders(self):
        with pytest.raises(ValueError, match="at least one order"):
            self.account.place_orders([])


class TestBasePlaceOrders:
    class _Venue(FuturesExchangeBase):
        pass

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        limit_order: FuturesLimitOrderAction,
        market_order: FuturesMarketOrderAction,
    ) -> None:
        self.venue = self._Venue()
        self.venue.place_limit_order = AsyncMock(  # type: ignore[method-assign]
            spec=FuturesExchangeBase.place_limit_order,
            return_value=PlacedOrder(order_id="limit"),
        )
        self.venue.place_market_order = AsyncMock(  # type: ignore[method-assign]
            spec=FuturesExchangeBase.place_market_order,
            return_value=PlacedOrder(order_id="market"),
        )
        account = FuturesAccount(self.venue, name="test")
        self.batch = account.place_orders([limit_order, market_order])

    async def test_venue_without_a_batch_endpoint(self):
        batch_result = await self.batch.execute()

        assert [o.placed_order.order_id for o in batch_result.orders] == [
            "limit",
            "market",
        ]
        self.venue.place_limit_order.assert_awaited_once()
        self.venue.place_market_order.assert_awaited_once()

    async def test_limit_order_keeps_its_time_in_force(self):
        await place_in_order(
            self.venue,
            [
                OrderRequest(
                    symbol=BTC_USDT,
                    side=OrderSide.BUY,
                    quantity=1.0,
                    limit_price=50000.0,
                    time_in_force=TimeInForce.POST_ONLY,
                )
            ],
        )

        assert (
            self.venue.place_limit_order.await_args.kwargs["time_in_force"]
            is TimeInForce.POST_ONLY
        )

    async def test_one_order_rejected_by_the_venue(self):
        self.venue.place_limit_order.side_effect = ExchangeRecoverableError("margin")

        batch_result = await self.batch.execute()

        (outcome,) = batch_result.orders
        assert outcome.placed_order.order_id == "market"

    async def test_transient_failure_fails_the_whole_batch(self):
        self.venue.place_limit_order.side_effect = ExchangeTransientError("timeout")

        with pytest.raises(ExchangeTransientError):
            await self.batch.execute()


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

        assert [order.order_id if order else None for order in placed] == ["one", None]

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


class TestModifyAction:
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        limit_order: FuturesLimitOrderAction,
        market_order: FuturesMarketOrderAction,
    ) -> None:
        self.exchange = Mock(spec=FuturesExchangeProtocol)
        self.account = FuturesAccount(self.exchange, name="test")
        self.limit_order = limit_order
        self.modify = self.account.modify_orders(
            [
                OrderModification(order_id="a", order=limit_order),
                OrderModification(order_id="b", order=market_order),
            ]
        )

    async def test_modifications_travel_as_one_request(self):
        self.exchange.modify_orders = AsyncMock(
            spec=FuturesExchangeProtocol.modify_orders,
            return_value=[PlacedOrder(order_id="a"), PlacedOrder(order_id="b2")],
        )

        await self.modify.execute()

        (requests,) = self.exchange.modify_orders.await_args.args
        assert [r.order_id for r in requests] == ["a", "b"]
        assert [r.order.limit_price for r in requests] == [50000.0, None]
        assert [r.order.quantity for r in requests] == [1.0, 2.0]

    async def test_each_replacement_keeps_its_own_follow_up(self):
        self.exchange.modify_orders = AsyncMock(
            spec=FuturesExchangeProtocol.modify_orders,
            return_value=[PlacedOrder(order_id="a"), PlacedOrder(order_id="b2")],
        )

        modify_result = await self.modify.execute()

        first, second = modify_result.orders
        assert first.placed_order.order_id == "a"
        assert first.on_filled is self.limit_order.on_filled
        assert second.on_filled is None
        assert first.placement.kind == "limit"
        assert second.placement.kind == "market"

    async def test_modification_the_venue_rejected(self):
        self.exchange.modify_orders = AsyncMock(
            spec=FuturesExchangeProtocol.modify_orders,
            return_value=[None, PlacedOrder(order_id="b2")],
        )

        modify_result = await self.modify.execute()

        (outcome,) = modify_result.orders
        assert outcome.placed_order.order_id == "b2"

    async def test_batch_without_modifications(self):
        with pytest.raises(ValueError, match="at least one order"):
            self.account.modify_orders([])

    async def test_builder_replacement_reshapes_into_the_order_it_builds(self):
        modify = self.account.modify_orders(
            [
                OrderModification(
                    order_id="a",
                    order=self.account.long_entry(BTC_USDT, 3.0).limit(49000.0),
                )
            ]
        )

        (replacement,) = modify.modifications
        assert replacement.order_id == "a"
        assert replacement.order.quantity == 3.0
        assert replacement.order.price == 49000.0

    async def test_unsized_builder_as_a_replacement(self):
        with pytest.raises(StrategyCriticalError, match="The order has no size"):
            self.account.modify_order("a", self.account.long_entry(BTC_USDT))

    async def test_one_order_reshaped_alone(self):
        self.exchange.modify_orders.return_value = [PlacedOrder(order_id="a")]
        modify_one = self.account.modify_order("a", self.limit_order)

        modify_result = await modify_one.execute()

        (request,) = self.exchange.modify_orders.await_args.args[0]
        assert request.order_id == "a"
        assert request.order.limit_price == 50000.0
        (outcome,) = modify_result.orders
        assert outcome.on_filled is self.limit_order.on_filled


class TestBaseModifyOrders:
    class _Venue(FuturesExchangeBase):
        pass

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        limit_order: FuturesLimitOrderAction,
        market_order: FuturesMarketOrderAction,
    ) -> None:
        self.venue = self._Venue()
        self.venue.get_open_orders = AsyncMock(  # type: ignore[method-assign]
            spec=FuturesExchangeBase.get_open_orders,
            return_value=[
                Mock(spec=OrderProtocol, order_id="a"),
                Mock(spec=OrderProtocol, order_id="b"),
            ],
        )
        self.venue.cancel_order_by_id = AsyncMock(  # type: ignore[method-assign]
            spec=FuturesExchangeBase.cancel_order_by_id
        )
        self.venue.place_limit_order = AsyncMock(  # type: ignore[method-assign]
            spec=FuturesExchangeBase.place_limit_order,
            return_value=PlacedOrder(order_id="limit"),
        )
        self.venue.place_market_order = AsyncMock(  # type: ignore[method-assign]
            spec=FuturesExchangeBase.place_market_order,
            return_value=PlacedOrder(order_id="market"),
        )
        account = FuturesAccount(self.venue, name="test")
        self.modify = account.modify_orders(
            [
                OrderModification(order_id="a", order=limit_order),
                OrderModification(order_id="b", order=market_order),
            ]
        )

    async def test_venue_without_a_modification_endpoint(self):
        modify_result = await self.modify.execute()

        assert [o.placed_order.order_id for o in modify_result.orders] == [
            "limit",
            "market",
        ]
        assert self.venue.cancel_order_by_id.await_count == 2

    async def test_target_no_longer_resting(self):
        self.venue.get_open_orders.return_value = [
            Mock(spec=OrderProtocol, order_id="b")
        ]

        modify_result = await self.modify.execute()

        (outcome,) = modify_result.orders
        assert outcome.placed_order.order_id == "market"
        self.venue.place_limit_order.assert_not_awaited()

    async def test_replacement_rejected_by_the_venue(self):
        self.venue.place_limit_order.side_effect = ExchangeRecoverableError("margin")

        modify_result = await self.modify.execute()

        (outcome,) = modify_result.orders
        assert outcome.placed_order.order_id == "market"

    async def test_transient_failure_fails_the_whole_batch(self):
        self.venue.place_market_order.side_effect = ExchangeTransientError("timeout")

        with pytest.raises(ExchangeTransientError):
            await self.modify.execute()


async def _on_filled(placed_order: PlacedOrder, fill: OrderFill | None) -> None:
    pass
