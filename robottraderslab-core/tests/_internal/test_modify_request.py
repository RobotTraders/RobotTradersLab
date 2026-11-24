from collections.abc import Callable
from dataclasses import replace

import pytest

from robottraderslab_core import Symbol
from robottraderslab_core._internal import BaseExchangeAction
from robottraderslab_core._internal.actions.members import carries_this_bot
from robottraderslab_core.exchanges import (
    OrderProtocol,
    OrderRequest,
    OrderSide,
    OrderType,
    StopLoss,
    TakeProfit,
    TimeInForce,
    TriggerType,
    modify_request,
    tag_of,
)

BTC = Symbol.create("BTC/USDT:USDT")
ORDER_ID = "order-7"
TAG = "1h-fast"


@pytest.fixture
def trigger_entry(
    create_resting_order: Callable[..., OrderProtocol],
) -> OrderProtocol:
    return create_resting_order(
        kind=OrderType.TRIGGER,
        side=OrderSide.BUY,
        limit_price=100.0,
        trigger_price=99.0,
        reduce_only=False,
        time_in_force=TimeInForce.GTC,
        trigger_type=TriggerType.MARK_PRICE,
        stop_loss_price=90.0,
        stop_loss_trigger_type=TriggerType.MARK_PRICE,
        take_profit_price=130.0,
        take_profit_trigger_type=TriggerType.LAST_PRICE,
    )


def test_a_price_alone_leaves_a_post_only_reduce_only_exit_as_it_rests(
    create_resting_order,
):
    resting = create_resting_order()

    request = modify_request(resting, limit_price=104.0)

    assert request.order_id == ORDER_ID
    assert replace(request.order, client_order_id=None) == OrderRequest(
        symbol=BTC,
        side=OrderSide.SELL,
        quantity=2.0,
        limit_price=104.0,
        time_in_force=TimeInForce.POST_ONLY,
        reduce_only=True,
    )


def test_a_partly_filled_order_keeps_its_whole_quantity_when_none_is_named(
    create_resting_order,
):
    resting = create_resting_order(quantity=2.0, remaining_quantity=0.5)

    request = modify_request(resting, limit_price=104.0)

    assert request.order.quantity == 2.0


def test_a_quantity_alone_replaces_the_size_with_the_filled_part_included(
    create_resting_order,
):
    resting = create_resting_order(quantity=2.0, remaining_quantity=1.5)

    request = modify_request(resting, quantity=3.0)

    assert request.order.quantity == 3.0
    assert request.order.limit_price == 105.0


def test_a_trigger_price_alone_leaves_the_trigger_type_the_limit_and_the_brackets(
    trigger_entry,
):
    request = modify_request(trigger_entry, trigger_price=98.0)

    assert replace(request.order, client_order_id=None) == OrderRequest(
        symbol=BTC,
        side=OrderSide.BUY,
        quantity=2.0,
        limit_price=100.0,
        time_in_force=TimeInForce.GTC,
        trigger_price=98.0,
        trigger_type=TriggerType.MARK_PRICE,
        stop_loss=StopLoss(trigger_price=90.0, trigger_type=TriggerType.MARK_PRICE),
        take_profit=TakeProfit(
            trigger_price=130.0, trigger_type=TriggerType.LAST_PRICE
        ),
    )


def test_a_stop_loss_alone_moves_its_level_and_leaves_both_trigger_types(
    trigger_entry,
):
    request = modify_request(trigger_entry, stop_loss=88.0)

    assert request.order.stop_loss == StopLoss(
        trigger_price=88.0, trigger_type=TriggerType.MARK_PRICE
    )
    assert request.order.take_profit == TakeProfit(
        trigger_price=130.0, trigger_type=TriggerType.LAST_PRICE
    )


def test_a_take_profit_alone_moves_its_level_and_leaves_the_stop_loss(
    trigger_entry,
):
    request = modify_request(trigger_entry, take_profit=140.0)

    assert request.order.take_profit == TakeProfit(
        trigger_price=140.0, trigger_type=TriggerType.LAST_PRICE
    )
    assert request.order.stop_loss == StopLoss(
        trigger_price=90.0, trigger_type=TriggerType.MARK_PRICE
    )


def test_a_stop_loss_named_on_an_order_without_one_adds_it_on_the_default_price(
    create_resting_order,
):
    resting = create_resting_order(kind=OrderType.LIMIT, side=OrderSide.BUY)

    request = modify_request(resting, stop_loss=90.0)

    assert request.order.stop_loss == StopLoss(
        trigger_price=90.0, trigger_type=TriggerType.DEFAULT
    )
    assert request.order.take_profit is None


def test_a_take_profit_named_on_an_order_without_one_adds_it_on_the_default_price(
    create_resting_order,
):
    resting = create_resting_order(kind=OrderType.LIMIT, side=OrderSide.BUY)

    request = modify_request(resting, take_profit=130.0)

    assert request.order.take_profit == TakeProfit(
        trigger_price=130.0, trigger_type=TriggerType.DEFAULT
    )
    assert request.order.stop_loss is None


def test_properties_the_read_states_none_for_take_the_request_defaults(
    create_resting_order,
):
    resting = create_resting_order(
        reduce_only=None,
        time_in_force=None,
        trigger_type=None,
        client_order_id=None,
    )

    request = modify_request(resting, limit_price=104.0)

    assert request.order.reduce_only is False
    assert request.order.time_in_force == TimeInForce.GTC
    assert request.order.trigger_type == TriggerType.DEFAULT
    assert request.order.client_order_id is None


def test_the_tag_of_the_client_order_id_stays_on_a_fresh_id(create_resting_order):
    resting = create_resting_order()

    request = modify_request(resting, limit_price=104.0)

    assert request.order.client_order_id != resting.client_order_id
    assert carries_this_bot(request.order.client_order_id)
    assert tag_of(request.order.client_order_id) == TAG


@pytest.mark.parametrize("foreign_id", ["placedbyhand", "other-1h-fast"])
def test_an_id_of_another_origin_derives_no_id(create_resting_order, foreign_id):
    resting = create_resting_order(client_order_id=foreign_id)

    request = modify_request(resting, limit_price=104.0)

    assert request.order.client_order_id is None


def test_an_untagged_id_of_this_bot_derives_a_fresh_untagged_id_of_this_bot(
    create_resting_order,
):
    untagged = BaseExchangeAction(symbol=BTC).client_order_id
    resting = create_resting_order(client_order_id=untagged)

    request = modify_request(resting, limit_price=104.0)

    assert request.order.client_order_id != untagged
    assert carries_this_bot(request.order.client_order_id)
    assert tag_of(request.order.client_order_id) is None


def test_a_market_trigger_keeps_no_limit_price(create_resting_order):
    resting = create_resting_order(
        kind=OrderType.TRIGGER, limit_price=None, trigger_price=99.0
    )

    request = modify_request(resting, trigger_price=98.0)

    assert request.order.limit_price is None
    assert request.order.trigger_price == 98.0


def test_a_call_naming_no_field(create_resting_order):
    resting = create_resting_order()

    with pytest.raises(ValueError, match=r"BTC/USDT:USDT.*order-7.*no field named"):
        modify_request(resting)


def test_a_limit_price_named_on_an_order_without_one(create_resting_order):
    resting = create_resting_order(
        kind=OrderType.TRIGGER, limit_price=None, trigger_price=99.0
    )

    with pytest.raises(ValueError, match=r"BTC/USDT:USDT.*order-7.*limit_price"):
        modify_request(resting, limit_price=100.0)


def test_a_trigger_price_named_on_an_order_without_one(create_resting_order):
    resting = create_resting_order()

    with pytest.raises(ValueError, match=r"BTC/USDT:USDT.*order-7.*trigger_price"):
        modify_request(resting, trigger_price=99.0)


@pytest.mark.parametrize(
    "kind",
    [
        OrderType.MARKET,
        OrderType.STOP_LOSS,
        OrderType.TAKE_PROFIT,
        OrderType.TRAILING_STOP,
        OrderType.LIQUIDATION,
    ],
)
@pytest.mark.parametrize(
    "field",
    ["quantity", "limit_price", "trigger_price", "stop_loss", "take_profit"],
)
def test_a_field_named_on_an_order_that_is_no_entry(create_resting_order, kind, field):
    resting = create_resting_order(kind=kind, limit_price=None)

    with pytest.raises(ValueError, match=rf"order-7.*{field}.*not reshaped"):
        modify_request(resting, **{field: 90.0})


@pytest.mark.parametrize("quantity", [1.0, 0.5])
def test_a_quantity_that_does_not_exceed_the_filled_part(
    create_resting_order, quantity
):
    resting = create_resting_order(quantity=2.0, remaining_quantity=1.0)

    with pytest.raises(ValueError, match=r"order-7.*quantity.*already filled"):
        modify_request(resting, quantity=quantity)


def test_a_quantity_above_the_filled_part_is_accepted(create_resting_order):
    resting = create_resting_order(quantity=2.0, remaining_quantity=1.0)

    request = modify_request(resting, quantity=1.5)

    assert request.order.quantity == 1.5


@pytest.mark.parametrize("missing", ["side", "quantity", "remaining_quantity"])
def test_an_order_whose_read_leaves_the_side_or_a_quantity_none(
    create_resting_order, missing
):
    resting = create_resting_order(**{missing: None})

    with pytest.raises(
        ValueError, match=r"order-7.*limit_price.*no side or no quantity"
    ):
        modify_request(resting, limit_price=104.0)


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_a_price_that_is_not_positive(create_resting_order, price):
    resting = create_resting_order()

    with pytest.raises(ValueError, match=r"order-7.*limit_price.*greater than 0"):
        modify_request(resting, limit_price=price)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
@pytest.mark.parametrize(
    "field", ["quantity", "limit_price", "trigger_price", "stop_loss", "take_profit"]
)
def test_a_non_finite_value(create_resting_order, field, value):
    resting = create_resting_order(trigger_price=99.0)

    with pytest.raises(ValueError, match=rf"order-7.*{field}.*{value} is not finite"):
        modify_request(resting, **{field: value})


def test_a_stop_loss_level_that_is_not_positive(
    create_resting_order,
):
    resting = create_resting_order()

    with pytest.raises(ValueError, match=r"order-7.*stop_loss.*greater than 0"):
        modify_request(resting, stop_loss=0.0)
