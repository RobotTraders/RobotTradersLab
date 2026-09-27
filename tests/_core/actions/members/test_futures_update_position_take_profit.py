from unittest.mock import AsyncMock, Mock

import pytest

from robottraderslab import Symbol
from robottraderslab._core import ActionResult
from robottraderslab.exceptions import ExchangeRecoverableError
from robottraderslab.exchanges import FuturesExchangeProtocol, PlacedOrder
from robottraderslab.futures.futures_update_position_take_profit import (
    UpdatePositionTakeProfitAction,
)


@pytest.fixture
def mock_exchange() -> FuturesExchangeProtocol:
    mock = Mock(spec=FuturesExchangeProtocol)
    mock.update_position_take_profit = AsyncMock(
        return_value=PlacedOrder(order_id="tp-1")
    )
    return mock


@pytest.fixture
def update_take_profit_action(mock_exchange) -> UpdatePositionTakeProfitAction:
    return UpdatePositionTakeProfitAction(
        exchange=mock_exchange,
        symbol=Symbol.create("BTC/USDT:USDT"),
        trigger_price=55_000.0,
    )


async def test_execute_returns_empty_action_result(update_take_profit_action):
    action_result = await update_take_profit_action.execute()

    assert isinstance(action_result, ActionResult)
    assert action_result.orders == ()


async def test_an_accepted_update_holds_the_order_the_venue_acknowledged(
    update_take_profit_action,
):
    await update_take_profit_action.execute()

    assert [placed.order_id for placed in update_take_profit_action.held] == ["tp-1"]


async def test_a_protection_refused_twice_is_not_held(update_take_profit_action):
    update_take_profit_action.exchange.update_position_take_profit.side_effect = (
        ExchangeRecoverableError("code 99999")
    )

    await update_take_profit_action.execute()

    assert update_take_profit_action.held == ()


async def test_exceptions_passthrough_execute(update_take_profit_action):
    error = ValueError("Test error")
    update_take_profit_action.exchange.update_position_take_profit.side_effect = error

    with pytest.raises(type(error)):
        await update_take_profit_action.execute()
