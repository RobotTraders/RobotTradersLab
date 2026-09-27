import logging
from datetime import datetime
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import (
    FuturesExchangeProtocol,
    OrderSide,
    PlacedOrder,
    PositionSide,
    PositionSnapshot,
    TimeInForce,
)
from robottraderslab.futures.futures_close_position import (
    ClosePositionAction,
    validate_closing_ratio,
)

BTC_USDT = Symbol.create("BTC/USDT:USDT")


def _held(side: PositionSide, quantity: float) -> dict[Symbol, PositionSnapshot]:
    return {
        BTC_USDT: PositionSnapshot(
            symbol=BTC_USDT,
            side=side,
            quantity=quantity,
            average_entry_price=100.0,
            entry_time=datetime(2026, 5, 1),
            leverage=1.0,
            liquidation_price=1.0,
        )
    }


@pytest.fixture
def venue() -> Mock:
    venue = Mock(spec=FuturesExchangeProtocol)
    venue.get_open_positions.return_value = _held(PositionSide.LONG, 2.0)
    venue.place_market_order.return_value = PlacedOrder(order_id="close-1")
    venue.place_limit_order.return_value = PlacedOrder(order_id="close-1")
    return venue


async def test_closes_at_market_what_the_venue_holds_when_it_runs(venue):
    close = ClosePositionAction(exchange=venue, symbol=BTC_USDT, reason="exit")

    await close.execute()

    venue.get_open_positions.assert_awaited_once_with([BTC_USDT])
    venue.place_market_order.assert_awaited_once_with(
        symbol=BTC_USDT,
        side=OrderSide.SELL,
        quantity=2.0,
        reduce_only=True,
        client_order_id=close.client_order_id,
        reason="exit",
        extra_fields=None,
    )


async def test_a_closing_ratio_closes_that_share_of_the_held_position(venue):
    close = ClosePositionAction(exchange=venue, symbol=BTC_USDT, closing_ratio=0.5)

    await close.execute()

    assert venue.place_market_order.await_args.kwargs["quantity"] == 1.0


async def test_a_short_is_closed_with_a_buy(venue):
    venue.get_open_positions.return_value = _held(PositionSide.SHORT, 2.0)
    close = ClosePositionAction(exchange=venue, symbol=BTC_USDT)

    await close.execute()

    assert venue.place_market_order.await_args.kwargs["side"] == OrderSide.BUY


async def test_a_limit_close_rests_reduce_only_at_its_price(venue):
    close = ClosePositionAction(
        exchange=venue,
        symbol=BTC_USDT,
        limit_price=120.0,
        time_in_force=TimeInForce.POST_ONLY,
    )

    await close.execute()

    venue.place_limit_order.assert_awaited_once_with(
        symbol=BTC_USDT,
        side=OrderSide.SELL,
        quantity=2.0,
        price=120.0,
        reduce_only=True,
        client_order_id=close.client_order_id,
        time_in_force=TimeInForce.POST_ONLY,
        reason=None,
        extra_fields=None,
    )
    venue.place_market_order.assert_not_awaited()


async def test_the_placement_reports_the_quantity_resolved_when_it_ran(venue):
    close = ClosePositionAction(
        exchange=venue, symbol=BTC_USDT, closing_ratio=0.5, limit_price=120.0
    )

    closed = await close.execute()

    placement = closed.orders[0].placement
    assert placement.quantity == 1.0
    assert placement.side == OrderSide.SELL
    assert placement.kind == "limit"
    assert placement.price == 120.0


async def test_a_flat_symbol_closes_nothing(venue, caplog):
    venue.get_open_positions.return_value = {}
    close = ClosePositionAction(exchange=venue, symbol=BTC_USDT)

    with caplog.at_level(logging.INFO):
        closed = await close.execute()

    assert closed.orders == ()
    venue.place_market_order.assert_not_awaited()
    assert [record.levelno for record in caplog.records] == [logging.INFO]
    assert "No position on `BTC/USDT:USDT` to close" in caplog.text


async def test_the_fill_callback_follows_the_placed_close(venue):
    async def on_filled(placed_order, fill):
        pass

    close = ClosePositionAction(exchange=venue, symbol=BTC_USDT, on_filled=on_filled)

    closed = await close.execute()

    assert closed.orders[0].on_filled is on_filled


@pytest.mark.parametrize("closing_ratio", [0.0, -0.5, 1.5, float("nan")])
def test_a_closing_ratio_outside_a_share(closing_ratio):
    with pytest.raises(ValueError, match="`closing_ratio` must be within"):
        validate_closing_ratio(closing_ratio)


@pytest.mark.parametrize("closing_ratio", [1e-9, 0.5, 1.0])
def test_a_closing_ratio_within_a_share(closing_ratio):
    validate_closing_ratio(closing_ratio)
