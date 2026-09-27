import asyncio
import logging
from collections.abc import Callable
from unittest.mock import AsyncMock, call

import pytest

from robottraderslab import Symbol
from robottraderslab._core import Notifications
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.exchanges import OrderFill, OrderPlacement, OrderSide


@pytest.fixture
def make_order_fill() -> Callable[[str], OrderFill]:
    def _make_order_fill(order_id: str) -> OrderFill:
        return OrderFill(
            order_id=order_id,
            symbol=Symbol.create("BTC/USDT"),
            side=OrderSide.BUY,
            quantity=1.0,
            kind="market",
        )

    return _make_order_fill


@pytest.fixture
def order_fill(make_order_fill) -> OrderFill:
    return make_order_fill("order-1")


async def test_a_recorded_fill_reaches_its_subscriber(order_fill):
    subscriber = AsyncMock()
    notifications = Notifications([subscriber])

    await notifications.record_fill(order_fill)
    await notifications.flush()

    subscriber.assert_awaited_once_with(order_fill)


async def test_every_subscriber_is_notified(order_fill):
    first = AsyncMock()
    second = AsyncMock()
    notifications = Notifications([first, second])

    await notifications.record_fill(order_fill)
    await notifications.flush()

    first.assert_awaited_once_with(order_fill)
    second.assert_awaited_once_with(order_fill)


async def test_events_reach_a_subscriber_in_booking_order_regardless_of_latency(
    make_order_fill,
):
    first_fill = make_order_fill("order-1")
    second_fill = make_order_fill("order-2")
    third_fill = make_order_fill("order-3")
    delays = iter([0.03, 0.01, 0.02])
    arrived = []

    async def subscriber(fill: OrderFill) -> None:
        await asyncio.sleep(next(delays))
        arrived.append(fill)

    notifications = Notifications([subscriber])
    await notifications.record_fill(first_fill)
    await notifications.record_fill(second_fill)
    await notifications.record_fill(third_fill)

    await notifications.flush()

    assert arrived == [first_fill, second_fill, third_fill]


async def test_two_subscribers_each_receive_every_recorded_fill(make_order_fill):
    first_fill = make_order_fill("order-1")
    second_fill = make_order_fill("order-2")
    first_subscriber = AsyncMock()
    second_subscriber = AsyncMock()
    notifications = Notifications([first_subscriber, second_subscriber])
    await notifications.record_fill(first_fill)
    await notifications.record_fill(second_fill)

    await notifications.flush()

    first_subscriber.assert_has_awaits([call(first_fill), call(second_fill)])
    second_subscriber.assert_has_awaits([call(first_fill), call(second_fill)])


async def test_flush_without_a_recorded_fill():
    notifications = Notifications([AsyncMock()])

    await notifications.flush()


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("boom"), StrategyCriticalError("boom")],
    ids=["exception", "critical"],
)
async def test_a_failing_subscriber_among_others(order_fill, caplog, failure):
    failing = AsyncMock(side_effect=failure)
    passing = AsyncMock()
    notifications = Notifications([failing, passing])
    await notifications.record_fill(order_fill)

    with caplog.at_level(logging.ERROR):
        await notifications.flush()

    passing.assert_awaited_once_with(order_fill)
    assert "boom" in caplog.text


async def test_a_failing_subscriber_does_not_block_its_later_events(make_order_fill):
    first_fill = make_order_fill("order-1")
    second_fill = make_order_fill("order-2")
    subscriber = AsyncMock(side_effect=[RuntimeError("boom"), None])
    notifications = Notifications([subscriber])
    await notifications.record_fill(first_fill)
    await notifications.record_fill(second_fill)

    await notifications.flush()

    subscriber.assert_any_await(second_fill)


@pytest.fixture
def placement(order_fill) -> OrderPlacement:
    return OrderPlacement(
        order_id="order-1",
        symbol=order_fill.symbol,
        side=OrderSide.BUY,
        quantity=1.0,
        kind="limit",
        price=100.0,
    )


async def test_a_recorded_placement_reaches_its_subscriber(placement):
    subscriber = AsyncMock()
    notifications = Notifications(on_placement=[subscriber])

    await notifications.record_placement(placement)
    await notifications.flush()

    subscriber.assert_awaited_once_with(placement)


async def test_a_fill_subscriber_hears_nothing_of_a_placement(placement):
    fill_subscriber = AsyncMock()
    notifications = Notifications(on_fill=[fill_subscriber])

    await notifications.record_placement(placement)
    await notifications.flush()

    fill_subscriber.assert_not_awaited()


async def test_a_placement_subscriber_hears_nothing_of_a_fill(order_fill):
    placement_subscriber = AsyncMock()
    notifications = Notifications(on_placement=[placement_subscriber])

    await notifications.record_fill(order_fill)
    await notifications.flush()

    placement_subscriber.assert_not_awaited()
