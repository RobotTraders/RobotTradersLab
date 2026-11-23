import logging
from unittest.mock import create_autospec

import pytest

from robottraderslab_core import Symbol
from robottraderslab_core._internal import ActionResult, OrderType
from robottraderslab_core.exceptions import (
    BotCriticalError,
    ExchangeRecoverableError,
    ExchangeTransientError,
    NoOpenPositionError,
)
from robottraderslab_core.exchanges import FuturesExchangeProtocol, PlacedOrder
from robottraderslab_core.futures import UpdatePositionTrailingStopAction

BTCUSDT = Symbol.create("BTC/USDT:USDT")
TRAIL_SHARE = 0.02


@pytest.fixture
def mock_exchange() -> FuturesExchangeProtocol:
    exchange = create_autospec(FuturesExchangeProtocol, instance=True)
    exchange.update_position_trailing_stop_ratio.return_value = PlacedOrder(
        order_id="trail-1"
    )
    return exchange


@pytest.fixture
def trail_without_one_resting(
    mock_exchange: FuturesExchangeProtocol,
) -> UpdatePositionTrailingStopAction:
    return UpdatePositionTrailingStopAction(
        exchange=mock_exchange,
        symbol=BTCUSDT,
        distance_ratio=TRAIL_SHARE,
        resting=(),
    )


async def test_the_share_of_the_price_is_set_on_the_symbol(
    trail_without_one_resting, mock_exchange
):
    action_result = await trail_without_one_resting.execute()

    mock_exchange.update_position_trailing_stop_ratio.assert_awaited_once_with(
        BTCUSDT, TRAIL_SHARE
    )
    assert action_result == ActionResult()


async def test_the_trailing_stop_placed_is_the_one_held(trail_without_one_resting):
    await trail_without_one_resting.execute()

    assert [placed.order_id for placed in trail_without_one_resting.held] == ["trail-1"]


async def test_a_trailing_stop_resting_is_kept_and_nothing_is_held(
    mock_exchange, create_resting_order
):
    resting = create_resting_order(kind=OrderType.TRAILING_STOP, trigger_price=None)
    action = UpdatePositionTrailingStopAction(
        exchange=mock_exchange,
        symbol=BTCUSDT,
        distance_ratio=TRAIL_SHARE,
        resting=(resting,),
    )

    await action.execute()

    mock_exchange.update_position_trailing_stop_ratio.assert_not_awaited()
    assert action.held == ()


async def test_a_symbol_with_no_position_is_dropped_at_info(
    trail_without_one_resting, mock_exchange, caplog
):
    mock_exchange.update_position_trailing_stop_ratio.side_effect = NoOpenPositionError(
        "No open position for symbol: BTC/USDT:USDT"
    )

    with caplog.at_level(logging.INFO):
        action_result = await trail_without_one_resting.execute()

    assert action_result == ActionResult()
    assert trail_without_one_resting.position_gone
    assert "trailing-stop update dropped" in caplog.text


async def test_a_rejected_trail_is_sent_once_more_and_logged(
    trail_without_one_resting, mock_exchange, caplog
):
    mock_exchange.update_position_trailing_stop_ratio.side_effect = [
        ExchangeRecoverableError("code 99999"),
        PlacedOrder(order_id="trail-2"),
    ]

    with caplog.at_level(logging.WARNING):
        await trail_without_one_resting.execute()

    assert [placed.order_id for placed in trail_without_one_resting.held] == ["trail-2"]
    assert "2.00% behind the price rejected, re-issuing it" in caplog.text


async def test_a_trail_rejected_twice_is_logged_at_error_and_held_by_none(
    trail_without_one_resting, mock_exchange, caplog
):
    mock_exchange.update_position_trailing_stop_ratio.side_effect = (
        ExchangeRecoverableError("code 99999")
    )

    with caplog.at_level(logging.ERROR):
        await trail_without_one_resting.execute()

    assert trail_without_one_resting.held == ()
    assert "trailing-stop does not hold at 2.00% behind the price" in caplog.text


async def test_a_transient_failure_is_left_to_the_caller(
    trail_without_one_resting, mock_exchange
):
    mock_exchange.update_position_trailing_stop_ratio.side_effect = (
        ExchangeTransientError("timeout")
    )

    with pytest.raises(ExchangeTransientError):
        await trail_without_one_resting.execute()


async def test_a_venue_offering_no_trailing_stop_refuses_the_booking(
    trail_without_one_resting, mock_exchange
):
    mock_exchange.update_position_trailing_stop_ratio.side_effect = BotCriticalError(
        "BTC/USDT:USDT: Hyperliquid offers no trailing stop"
    )

    with pytest.raises(BotCriticalError, match="offers no trailing stop"):
        await trail_without_one_resting.execute()

    assert trail_without_one_resting.held == ()
