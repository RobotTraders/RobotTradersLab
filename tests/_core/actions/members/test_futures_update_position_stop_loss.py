import logging
from unittest.mock import AsyncMock, Mock

import pytest

from robottraderslab import Symbol
from robottraderslab._core import ActionResult
from robottraderslab.exceptions import (
    ExchangeRecoverableError,
    ExchangeTransientError,
    NoOpenPositionError,
)
from robottraderslab.exchanges import FuturesExchangeProtocol, PlacedOrder
from robottraderslab.futures.futures_update_position_stop_loss import (
    UpdatePositionStopLossAction,
)


@pytest.fixture
def mock_exchange() -> FuturesExchangeProtocol:
    mock = Mock(spec=FuturesExchangeProtocol)
    mock.update_position_stop_loss = AsyncMock(
        return_value=PlacedOrder(order_id="sl-1")
    )
    return mock


@pytest.fixture
def update_stop_loss_action(mock_exchange) -> UpdatePositionStopLossAction:
    return UpdatePositionStopLossAction(
        exchange=mock_exchange,
        symbol=Symbol.create("BTC/USDT:USDT"),
        trigger_price=42_000.0,
    )


async def test_execute_returns_empty_action_result(update_stop_loss_action):
    action_result = await update_stop_loss_action.execute()

    assert isinstance(action_result, ActionResult)
    assert action_result.orders == ()


async def test_exceptions_passthrough_execute(update_stop_loss_action):
    error = ValueError("Test error")
    update_stop_loss_action.exchange.update_position_stop_loss.side_effect = error

    with pytest.raises(type(error)):
        await update_stop_loss_action.execute()


async def test_an_accepted_update_holds_the_order_the_venue_acknowledged(
    update_stop_loss_action,
):
    await update_stop_loss_action.execute()

    assert [placed.order_id for placed in update_stop_loss_action.held] == ["sl-1"]


async def test_an_accepted_update_says_the_position_stands(update_stop_loss_action):
    await update_stop_loss_action.execute()

    assert update_stop_loss_action.position_gone is False


async def test_a_symbol_with_no_position_left_says_the_position_is_gone(
    update_stop_loss_action,
):
    update_stop_loss_action.exchange.update_position_stop_loss.side_effect = (
        NoOpenPositionError("No open position on BTC/USDT:USDT")
    )

    await update_stop_loss_action.execute()

    assert update_stop_loss_action.position_gone is True


async def test_a_protection_refused_twice_is_not_held(update_stop_loss_action):
    update_stop_loss_action.exchange.update_position_stop_loss.side_effect = (
        ExchangeRecoverableError("code 99999")
    )

    await update_stop_loss_action.execute()

    assert update_stop_loss_action.held == ()


async def test_a_symbol_with_no_position_left_drops_the_update(
    update_stop_loss_action, caplog
):
    update_stop_loss_action.exchange.update_position_stop_loss.side_effect = (
        NoOpenPositionError("No open position on BTC/USDT:USDT")
    )

    with caplog.at_level(logging.INFO):
        await update_stop_loss_action.execute()

    assert update_stop_loss_action.exchange.update_position_stop_loss.await_count == 1
    assert update_stop_loss_action.held == ()
    assert [record.levelno for record in caplog.records] == [logging.INFO]


async def test_a_rejected_update_is_sent_once_more(update_stop_loss_action):
    update_stop_loss_action.exchange.update_position_stop_loss.side_effect = [
        ExchangeRecoverableError("code 99999"),
        None,
    ]

    await update_stop_loss_action.execute()

    assert update_stop_loss_action.exchange.update_position_stop_loss.await_count == 2


async def test_a_single_rejection_names_the_symbol_the_protection_and_the_level(
    update_stop_loss_action, caplog
):
    update_stop_loss_action.exchange.update_position_stop_loss.side_effect = [
        ExchangeRecoverableError("code 99999"),
        None,
    ]

    with caplog.at_level(logging.WARNING):
        await update_stop_loss_action.execute()

    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert "BTC/USDT:USDT" in record.message
    assert "stop-loss" in record.message
    assert "42000.0" in record.message


async def test_a_protection_rejected_twice_names_the_symbol_the_protection_and_the_level(
    update_stop_loss_action, caplog
):
    update_stop_loss_action.exchange.update_position_stop_loss.side_effect = (
        ExchangeRecoverableError("code 99999")
    )

    with caplog.at_level(logging.WARNING):
        await update_stop_loss_action.execute()

    [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert "BTC/USDT:USDT" in record.message
    assert "stop-loss" in record.message
    assert "42000.0" in record.message


async def test_a_protection_rejected_twice_is_not_sent_a_third_time(
    update_stop_loss_action,
):
    update_stop_loss_action.exchange.update_position_stop_loss.side_effect = (
        ExchangeRecoverableError("code 99999")
    )

    await update_stop_loss_action.execute()

    assert update_stop_loss_action.exchange.update_position_stop_loss.await_count == 2


async def test_a_transient_rejection_of_the_second_attempt_is_left_to_the_executor_retry(
    update_stop_loss_action,
):
    update_stop_loss_action.exchange.update_position_stop_loss.side_effect = [
        ExchangeRecoverableError("code 99999"),
        ExchangeTransientError("503"),
    ]

    with pytest.raises(ExchangeTransientError):
        await update_stop_loss_action.execute()
