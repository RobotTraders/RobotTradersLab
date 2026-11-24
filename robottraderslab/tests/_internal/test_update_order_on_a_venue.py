from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from robottraderslab import Symbol
from robottraderslab._internal import NO_ORDER_OVERRIDES, OrderOverrides, OrderSettings
from robottraderslab.exceptions import BotCriticalError
from robottraderslab.exchanges import (
    FuturesExchangeProtocol,
    OrderModifyRequest,
    OrderProtocol,
    OrderSide,
    OrderType,
    PlacedOrder,
    StopLoss,
    TakeProfit,
    TimeInForce,
    TriggerType,
    client_order_id_carrying,
    tag_of,
)
from robottraderslab.futures import FuturesAccount
from robottraderslab_core.engine import carries_this_bot
from robottraderslab_core.exchanges import next_client_order_id

BTC = Symbol.create("BTC/USDT:USDT")
ORDER_ID = "order-7"

PLAIN_BUY = {
    "side": OrderSide.BUY,
    "quantity": 6.0,
    "remaining_quantity": 6.0,
    "limit_price": 10.0,
    "reduce_only": False,
    "time_in_force": TimeInForce.GTC,
    "client_order_id": None,
}


@pytest.fixture
def create_plain_buy(
    create_resting_order: Callable[..., OrderProtocol],
) -> Callable[..., OrderProtocol]:
    def create(**fields: Any) -> OrderProtocol:
        return create_resting_order(**{**PLAIN_BUY, **fields})

    return create


@pytest.fixture
def venue(make_venue: Callable[..., Mock]) -> Mock:
    exchange = make_venue(quantity_step=1.0, minimum_order_value=5.0)
    exchange.placement_reserve_rate = 0.0

    async def replaced(requests: list[OrderModifyRequest]) -> list[PlacedOrder]:
        return [PlacedOrder(order_id=f"new-{index}") for index in range(len(requests))]

    exchange.modify_orders = AsyncMock(
        spec=FuturesExchangeProtocol.modify_orders, side_effect=replaced
    )
    return exchange


@pytest.fixture
def create_account(venue: Mock) -> Callable[..., Awaitable[FuturesAccount]]:
    async def create(
        order_overrides: OrderOverrides = NO_ORDER_OVERRIDES,
    ) -> FuturesAccount:
        account = FuturesAccount(venue, order_overrides=order_overrides)
        await account._read_instruments([BTC])
        return account

    return create


def sent_request(venue: Mock) -> OrderModifyRequest:
    (requests,) = venue.modify_orders.await_args.args
    (request,) = requests
    return request


async def test_a_reduce_only_post_only_exit_moved_by_price_alone_is_sent_as_it_rests(
    create_account, create_resting_order, venue
):
    account = await create_account()
    resting = create_resting_order(
        kind=OrderType.LIMIT,
        reduce_only=True,
        time_in_force=TimeInForce.POST_ONLY,
    )

    await account.update_order(resting, limit_price=104.0).execute()

    request = sent_request(venue)
    assert request.order.limit_price == 104.0
    assert request.order.reduce_only is True
    assert request.order.time_in_force == TimeInForce.POST_ONLY


async def test_a_trigger_entry_read_with_another_time_in_force_rests_gtc(
    create_account, create_plain_buy, venue
):
    account = await create_account()
    resting = create_plain_buy(
        kind=OrderType.TRIGGER,
        limit_price=100.0,
        trigger_price=101.0,
        trigger_type=TriggerType.MARK_PRICE,
        time_in_force=TimeInForce.IOC,
    )

    await account.update_order(resting, limit_price=99.0).execute()

    assert sent_request(venue).order.time_in_force == TimeInForce.GTC


async def test_an_order_on_a_symbol_never_declared_for_the_account(
    create_plain_buy, venue
):
    account = FuturesAccount(venue)
    resting = create_plain_buy()

    with pytest.raises(BotCriticalError, match="BTC"):
        account.update_order(resting, limit_price=11.0)


@pytest.mark.parametrize("tag", ["tp-reached", "sl"])
async def test_an_order_carrying_an_engine_tag(create_account, create_plain_buy, tag):
    account = await create_account()
    resting = create_plain_buy(client_order_id=client_order_id_carrying(tag))

    with pytest.raises(ValueError, match=tag):
        account.update_order(resting, limit_price=11.0)


@pytest.mark.parametrize(
    "resting_fields",
    [
        {"kind": OrderType.LIMIT},
        {
            "kind": OrderType.TRIGGER,
            "limit_price": None,
            "trigger_price": 101.0,
            "trigger_type": TriggerType.MARK_PRICE,
        },
    ],
    ids=["a limit order", "a trigger order with no limit"],
)
@pytest.mark.parametrize("client_order_id", [None, "placed-by-hand"])
async def test_an_order_this_bot_did_not_place_is_sent_with_no_client_order_id(
    create_account, create_plain_buy, venue, resting_fields, client_order_id
):
    account = await create_account()
    resting = create_plain_buy(client_order_id=client_order_id, **resting_fields)

    await account.update_order(resting, quantity=7.0).execute()

    assert sent_request(venue).order.client_order_id is None


async def test_an_order_of_this_bot_is_sent_with_a_fresh_id_carrying_its_tag(
    create_account, create_plain_buy, venue
):
    account = await create_account()
    resting_id = next_client_order_id("1h-fast")
    resting = create_plain_buy(client_order_id=resting_id)

    await account.update_order(resting, limit_price=11.0).execute()

    sent_id = sent_request(venue).order.client_order_id
    assert carries_this_bot(sent_id)
    assert tag_of(sent_id) == "1h-fast"
    assert sent_id != resting_id


async def test_the_configured_bracket_trigger_types_are_not_applied_again(
    create_account, create_plain_buy, venue
):
    account = await create_account(
        OrderOverrides(
            stop_loss_trigger_type=TriggerType.MARK_PRICE,
            take_profit_trigger_type=TriggerType.MARK_PRICE,
        )
    )
    resting = create_plain_buy(
        stop_loss_price=9.0,
        stop_loss_trigger_type=TriggerType.LAST_PRICE,
        take_profit_price=12.0,
        take_profit_trigger_type=TriggerType.LAST_PRICE,
    )

    await account.update_order(resting, stop_loss=8.0).execute()

    request = sent_request(venue).order
    assert request.stop_loss == StopLoss(
        trigger_price=8.0, trigger_type=TriggerType.LAST_PRICE
    )
    assert request.take_profit == TakeProfit(
        trigger_price=12.0, trigger_type=TriggerType.LAST_PRICE
    )


async def test_the_configured_entry_time_in_force_is_not_applied_again(
    create_account, create_plain_buy, venue
):
    account = await create_account(
        OrderOverrides(entry=OrderSettings(time_in_force=TimeInForce.IOC))
    )
    resting = create_plain_buy(reduce_only=False, time_in_force=TimeInForce.GTC)

    await account.update_order(resting, limit_price=11.0).execute()

    assert sent_request(venue).order.time_in_force == TimeInForce.GTC


async def test_the_configured_exit_time_in_force_is_not_applied_again(
    create_account, create_plain_buy, venue
):
    account = await create_account(
        OrderOverrides(exit=OrderSettings(time_in_force=TimeInForce.POST_ONLY))
    )
    resting = create_plain_buy(
        side=OrderSide.SELL, reduce_only=True, time_in_force=TimeInForce.GTC
    )

    await account.update_order(resting, limit_price=11.0).execute()

    assert sent_request(venue).order.time_in_force == TimeInForce.GTC


async def test_the_configured_trigger_type_is_not_applied_again(
    create_account, create_plain_buy, venue
):
    account = await create_account(
        OrderOverrides(entry=OrderSettings(trigger_type=TriggerType.INDEX_PRICE))
    )
    resting = create_plain_buy(
        kind=OrderType.TRIGGER,
        limit_price=None,
        trigger_price=101.0,
        trigger_type=TriggerType.LAST_PRICE,
    )

    await account.update_order(resting, trigger_price=102.0).execute()

    assert sent_request(venue).order.trigger_type == TriggerType.LAST_PRICE


async def test_a_quantity_between_steps_is_rounded_down_to_a_step(
    create_account, create_plain_buy, venue
):
    account = await create_account()
    resting = create_plain_buy(quantity=6.0, remaining_quantity=6.0)

    await account.update_order(resting, quantity=8.7).execute()

    assert sent_request(venue).order.quantity == 8.0


async def test_a_quantity_under_the_minimum_order_value_cancels_the_resting_order(
    create_account, create_plain_buy, venue
):
    account = await create_account()
    resting = create_plain_buy(quantity=6.0, remaining_quantity=6.0, limit_price=1.0)

    action_result = await account.update_order(resting, quantity=4.0).execute()

    venue.cancel_order_by_id.assert_awaited_once_with(BTC, ORDER_ID)
    venue.modify_orders.assert_not_awaited()
    assert action_result.orders == ()
