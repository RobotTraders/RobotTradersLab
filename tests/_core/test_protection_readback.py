import logging
from collections.abc import Callable
from unittest.mock import AsyncMock, Mock

import pytest

from robottraderslab import Symbol
from robottraderslab._core import confirm_booked_protections
from robottraderslab.exceptions import ExchangeRecoverableError, NoOpenPositionError
from robottraderslab.exchanges import FuturesExchangeProtocol, PlacedOrder
from robottraderslab.strategies import OrderProtocol
from robottraderslab.strategies.futures import (
    UpdatePositionStopLossAction,
    UpdatePositionTakeProfitAction,
)

DOGE = Symbol.create("DOGE/USDT:USDT")
BTC = Symbol.create("BTC/USDT:USDT")
STOP_LOSS = 0.2
STOP_LOSS_ID = "sl-1"
TAKE_PROFIT = 0.3


@pytest.fixture
def create_resting_order() -> Callable[[Symbol, str, str], Mock]:
    def _create_resting_order(symbol: Symbol, kind: str, order_id: str) -> Mock:
        order = Mock(spec=OrderProtocol)
        order.symbol = symbol
        order.kind = kind
        order.order_id = order_id
        return order

    return _create_resting_order


@pytest.fixture
def create_venue() -> Callable[..., Mock]:
    def _create_venue(*resting: Mock, acknowledges: str = STOP_LOSS_ID) -> Mock:
        venue = Mock(spec=FuturesExchangeProtocol)
        venue.get_open_orders = AsyncMock(return_value=list(resting))
        venue.update_position_stop_loss = AsyncMock(
            return_value=PlacedOrder(order_id=acknowledges)
        )
        return venue

    return _create_venue


@pytest.fixture
def create_closed_take_profit() -> Callable[
    [Mock, Symbol], UpdatePositionTakeProfitAction
]:
    async def _create_closed_take_profit(
        venue: Mock, symbol: Symbol
    ) -> UpdatePositionTakeProfitAction:
        venue.update_position_take_profit = AsyncMock(
            side_effect=NoOpenPositionError("position closed")
        )
        action = UpdatePositionTakeProfitAction(
            exchange=venue, symbol=symbol, trigger_price=TAKE_PROFIT
        )
        await action.execute()
        return action

    return _create_closed_take_profit


@pytest.fixture
def create_held_stop_loss() -> Callable[[Mock, Symbol], UpdatePositionStopLossAction]:
    async def _create_held_stop_loss(
        venue: Mock, symbol: Symbol
    ) -> UpdatePositionStopLossAction:
        action = UpdatePositionStopLossAction(
            exchange=venue, symbol=symbol, trigger_price=STOP_LOSS
        )
        await action.execute()
        return action

    return _create_held_stop_loss


async def test_each_venue_is_read_once_for_its_own_protections(
    create_resting_order, create_venue, create_held_stop_loss
):
    first_venue = create_venue(create_resting_order(DOGE, "stop-loss", STOP_LOSS_ID))
    second_venue = create_venue(create_resting_order(BTC, "stop-loss", STOP_LOSS_ID))
    protections = [
        await create_held_stop_loss(first_venue, DOGE),
        await create_held_stop_loss(second_venue, BTC),
    ]

    await confirm_booked_protections(protections)

    first_venue.get_open_orders.assert_awaited_once_with([DOGE])
    second_venue.get_open_orders.assert_awaited_once_with([BTC])


async def test_a_protection_the_venue_refused_is_not_read_back(
    create_venue, create_held_stop_loss
):
    venue = create_venue()
    venue.update_position_stop_loss.side_effect = ExchangeRecoverableError("99999")
    refused = await create_held_stop_loss(venue, DOGE)

    await confirm_booked_protections([refused])

    venue.get_open_orders.assert_not_awaited()


async def test_an_order_under_another_id_does_not_hold_the_protection(
    create_resting_order, create_venue, create_held_stop_loss
):
    venue = create_venue(create_resting_order(DOGE, "stop-loss", "sl-old"))
    protection = await create_held_stop_loss(venue, DOGE)

    await confirm_booked_protections([protection])

    assert venue.update_position_stop_loss.await_count == 2


async def test_only_the_last_update_of_a_protection_is_confirmed(
    create_resting_order, create_venue, create_held_stop_loss
):
    venue = create_venue(create_resting_order(DOGE, "stop-loss", "sl-2"))
    first = await create_held_stop_loss(venue, DOGE)
    venue.update_position_stop_loss.return_value = PlacedOrder(order_id="sl-2")
    second = UpdatePositionStopLossAction(
        exchange=venue, symbol=DOGE, trigger_price=0.3
    )
    await second.execute()

    await confirm_booked_protections([first, second])

    assert venue.update_position_stop_loss.await_count == 2


async def test_a_re_issue_the_venue_refuses_is_not_read_back_again(
    create_venue, create_held_stop_loss
):
    venue = create_venue()
    protection = await create_held_stop_loss(venue, DOGE)
    venue.update_position_stop_loss.side_effect = ExchangeRecoverableError("99999")

    await confirm_booked_protections([protection])

    venue.get_open_orders.assert_awaited_once()


async def test_a_read_the_venue_rejects_leaves_the_cycle_running(
    create_venue, create_held_stop_loss, caplog
):
    venue = create_venue()
    venue.get_open_orders.side_effect = ExchangeRecoverableError("busy")
    protection = await create_held_stop_loss(venue, DOGE)

    with caplog.at_level(logging.WARNING):
        await confirm_booked_protections([protection])

    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert "DOGE/USDT:USDT" in record.message
    assert venue.update_position_stop_loss.await_count == 1


async def test_a_protection_whose_position_the_cycle_closed_is_not_read_back(
    create_venue, create_held_stop_loss, create_closed_take_profit, caplog
):
    venue = create_venue()
    protections = [
        await create_held_stop_loss(venue, DOGE),
        await create_closed_take_profit(venue, DOGE),
    ]

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        await confirm_booked_protections(protections)

    venue.get_open_orders.assert_not_awaited()
    assert venue.update_position_stop_loss.await_count == 1
    assert caplog.records == []


async def test_a_closed_position_leaves_another_symbol_read_back(
    create_resting_order, create_venue, create_held_stop_loss, create_closed_take_profit
):
    venue = create_venue(create_resting_order(BTC, "stop-loss", STOP_LOSS_ID))
    protections = [
        await create_held_stop_loss(venue, DOGE),
        await create_closed_take_profit(venue, DOGE),
        await create_held_stop_loss(venue, BTC),
    ]

    await confirm_booked_protections(protections)

    venue.get_open_orders.assert_awaited_once_with([BTC])


async def test_a_protection_missing_while_the_position_stands_is_reported(
    create_resting_order, create_venue, create_held_stop_loss, caplog
):
    venue = create_venue(create_resting_order(DOGE, "stop-loss", "sl-old"))
    protection = await create_held_stop_loss(venue, DOGE)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await confirm_booked_protections([protection])

    [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "DOGE/USDT:USDT" in warning.message
    assert "stop-loss" in warning.message
    assert str(STOP_LOSS) in warning.message
