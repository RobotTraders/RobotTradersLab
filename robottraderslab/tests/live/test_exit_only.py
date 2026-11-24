from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import (
    FuturesExchangeProtocol,
    MarginMode,
    OrderProtocol,
    OrderSide,
    OrderType,
    PositionSide,
    PositionSnapshot,
)
from robottraderslab.futures import (
    CancelOrderByIdAction,
    CancelOrdersForSymbolAction,
    ClosePositionAction,
    FuturesLimitOrderAction,
    FuturesMarketOrderAction,
    FuturesOrderBatchAction,
    FuturesOrderModifyAction,
    SetLeverageAction,
    SetMarginModeAction,
    UpdatePositionStopLossAction,
    UpdatePositionTakeProfitAction,
    UpdatePositionTrailingStopAction,
)
from robottraderslab.live.exit_only import ExitOnlyDecision, restrict_symbol
from robottraderslab_core.engine import OrderReplacement, to_step_count
from robottraderslab_core.exchanges import BaseExchangeAction, StopLoss, TakeProfit

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")
OTHER_BOTS_ID = "aaaaaa000000-1-grid"


@pytest.fixture
def venue() -> Mock:
    return Mock(spec=FuturesExchangeProtocol)


@pytest.fixture
def make_market(venue: Mock) -> Callable[..., FuturesMarketOrderAction]:
    def make(
        side: OrderSide = OrderSide.SELL,
        quantity: float = 1.0,
        symbol: Symbol = BTC,
        **fields,
    ) -> FuturesMarketOrderAction:
        return FuturesMarketOrderAction(
            exchange=venue, symbol=symbol, side=side, quantity=quantity, **fields
        )

    return make


@pytest.fixture
def make_limit(venue: Mock) -> Callable[..., FuturesLimitOrderAction]:
    def make(
        side: OrderSide = OrderSide.SELL,
        quantity: float = 1.0,
        symbol: Symbol = BTC,
        price: float = 100.0,
        **fields,
    ) -> FuturesLimitOrderAction:
        return FuturesLimitOrderAction(
            exchange=venue,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            **fields,
        )

    return make


def _position(quantity: float, side: PositionSide = PositionSide.LONG):
    return PositionSnapshot(
        symbol=BTC,
        side=side,
        quantity=quantity,
        average_entry_price=100.0,
        entry_time=datetime(2026, 1, 1, tzinfo=UTC),
        leverage=1.0,
        liquidation_price=None,
        stop_loss_price=None,
        take_profit_price=None,
    )


def _decide(
    actions: Sequence[BaseExchangeAction],
    position: PositionSnapshot | None = None,
    open_orders: Sequence[OrderProtocol] = (),
    *,
    labels_resting_triggers: bool = True,
) -> ExitOnlyDecision:
    return restrict_symbol(
        BTC,
        actions,
        position,
        open_orders,
        labels_resting_triggers=labels_resting_triggers,
    )


class TestSettings:
    @pytest.mark.parametrize(
        "protect",
        [
            lambda venue: UpdatePositionStopLossAction(
                exchange=venue, symbol=BTC, trigger_price=95.0
            ),
            lambda venue: UpdatePositionTakeProfitAction(
                exchange=venue, symbol=BTC, trigger_price=110.0
            ),
            lambda venue: UpdatePositionTrailingStopAction(
                exchange=venue, symbol=BTC, distance_ratio=0.02, resting=()
            ),
        ],
        ids=["stop-loss", "take-profit", "trailing-stop"],
    )
    def test_a_protection_move_is_kept(self, venue, protect):
        move = protect(venue)

        decision = _decide([move], _position(1.0))

        assert decision.actions == (move,)

    def test_a_leverage_setting_is_kept(self, venue):
        setting = SetLeverageAction(exchange=venue, symbol=BTC, leverage=3.0)

        decision = _decide([setting])

        assert decision.actions == (setting,)

    def test_a_margin_mode_setting_is_kept(self, venue):
        setting = SetMarginModeAction(
            exchange=venue, symbol=BTC, margin_mode=MarginMode.ISOLATED
        )

        decision = _decide([setting])

        assert decision.actions == (setting,)

    def test_an_action_of_a_class_exit_only_mode_does_not_know_is_dropped(self):
        unknown = BaseExchangeAction(symbol=BTC)

        decision = _decide([unknown], _position(1.0))

        assert decision.actions == ()
        assert decision.dropped == (unknown,)

    def test_an_action_on_another_symbol_is_returned_as_booked(self, make_market):
        entry = make_market(OrderSide.BUY, symbol=ETH)

        decision = _decide([entry], _position(1.0))

        assert decision.actions == (entry,)
        assert decision.dropped == ()


class TestMarketOrders:
    def test_an_order_closing_within_the_position_is_kept_as_booked(self, make_market):
        exit_order = make_market(OrderSide.SELL, 1.0)

        decision = _decide([exit_order], _position(1.0))

        assert decision.actions == (exit_order,)
        assert decision.actions[0] is exit_order
        assert decision.cut == ()

    def test_an_order_closing_a_short_is_kept(self, make_market):
        exit_order = make_market(OrderSide.BUY, 1.0)

        decision = _decide([exit_order], _position(1.0, PositionSide.SHORT))

        assert decision.actions == (exit_order,)

    def test_an_order_closing_within_the_position_goes_without_its_protections(
        self, make_market
    ):
        exit_order = make_market(
            OrderSide.SELL,
            0.5,
            stop_loss=StopLoss(trigger_price=90.0),
            take_profit=TakeProfit(trigger_price=120.0),
        )

        decision = _decide([exit_order], _position(1.0))

        sent = decision.actions[0]
        assert sent.stop_loss is None
        assert sent.take_profit is None
        assert sent.quantity == 0.5
        assert decision.cut == ()

    def test_the_booked_order_keeps_its_protections_when_a_copy_is_sent(
        self, make_market
    ):
        stop_loss = StopLoss(trigger_price=90.0)
        exit_order = make_market(OrderSide.SELL, 0.5, stop_loss=stop_loss)

        _decide([exit_order], _position(1.0))

        assert exit_order.stop_loss is stop_loss

    def test_an_order_larger_than_the_position_is_cut_to_it(self, make_market):
        exit_order = make_market(
            OrderSide.SELL, 3.0, stop_loss=StopLoss(trigger_price=90.0)
        )

        decision = _decide([exit_order], _position(1.0))

        sent = decision.actions[0]
        assert sent.quantity == pytest.approx(1.0)
        assert sent.stop_loss is None
        assert decision.cut[0].original is exit_order
        assert decision.cut[0].quantity == pytest.approx(1.0)

    def test_the_cut_order_keeps_the_id_the_booking_gave_it(self, make_market):
        exit_order = make_market(OrderSide.SELL, 3.0, tag="1h-exit")

        decision = _decide([exit_order], _position(1.0))

        assert decision.actions[0].id == exit_order.id
        assert decision.actions[0].client_order_id == exit_order.client_order_id

    def test_the_booked_order_is_left_at_its_quantity_when_it_is_cut(self, make_market):
        exit_order = make_market(OrderSide.SELL, 3.0)

        _decide([exit_order], _position(1.0))

        assert exit_order.quantity == 3.0

    def test_a_second_order_after_the_first_used_the_position_up_is_dropped(
        self, make_market
    ):
        first = make_market(OrderSide.SELL, 1.0)
        second = make_market(OrderSide.SELL, 1.0)

        decision = _decide([first, second], _position(1.0))

        assert decision.actions == (first,)
        assert decision.dropped == (second,)

    def test_the_second_of_two_orders_is_cut_to_what_the_first_left(self, make_market):
        first = make_market(OrderSide.SELL, 0.6)
        second = make_market(OrderSide.SELL, 0.6)

        decision = _decide([first, second], _position(1.0))

        assert decision.actions[1].quantity == pytest.approx(0.4)
        assert decision.cut[0].original is second

    def test_an_order_on_the_positions_side_is_dropped(self, make_market):
        entry = make_market(OrderSide.BUY, 1.0)

        decision = _decide([entry], _position(1.0))

        assert decision.actions == ()
        assert decision.dropped == (entry,)

    def test_an_entry_carrying_a_stop_loss_is_dropped_whole(self, make_market):
        entry = make_market(OrderSide.BUY, 1.0, stop_loss=StopLoss(trigger_price=90.0))

        decision = _decide([entry], _position(1.0))

        assert decision.actions == ()

    def test_every_order_on_a_symbol_holding_no_position_is_dropped(self, make_market):
        entry = make_market(OrderSide.BUY, 1.0)
        exit_order = make_market(OrderSide.SELL, 1.0)

        decision = _decide([entry, exit_order], None)

        assert decision.actions == ()
        assert decision.dropped == (entry, exit_order)

    def test_a_reduce_only_order_is_kept_on_the_positions_side(self, make_market):
        flagged = make_market(OrderSide.BUY, 5.0, reduce_only=True)

        decision = _decide([flagged], _position(1.0))

        assert decision.actions == (flagged,)

    def test_a_reduce_only_order_is_kept_on_a_symbol_holding_no_position(
        self, make_market
    ):
        flagged = make_market(OrderSide.SELL, 5.0, reduce_only=True)

        decision = _decide([flagged], None)

        assert decision.actions == (flagged,)

    def test_a_reduce_only_order_closing_the_position_uses_it_up(self, make_market):
        flagged = make_market(OrderSide.SELL, 1.0, reduce_only=True)
        exit_order = make_market(OrderSide.SELL, 1.0)

        decision = _decide([flagged, exit_order], _position(1.0))

        assert decision.dropped == (exit_order,)

    def test_a_reduce_only_order_on_the_positions_side_uses_nothing_up(
        self, make_market
    ):
        flagged = make_market(OrderSide.BUY, 5.0, reduce_only=True)
        exit_order = make_market(OrderSide.SELL, 1.0)

        decision = _decide([flagged, exit_order], _position(1.0))

        assert decision.actions == (flagged, exit_order)


class TestCloses:
    def test_a_close_is_kept(self, venue):
        close = ClosePositionAction(exchange=venue, symbol=BTC)

        decision = _decide([close], _position(1.0))

        assert decision.actions == (close,)

    def test_a_close_kept_on_a_symbol_holding_no_position(self, venue):
        close = ClosePositionAction(exchange=venue, symbol=BTC)

        decision = _decide([close], None)

        assert decision.actions == (close,)

    def test_a_close_of_the_whole_position_uses_it_up(self, venue, make_market):
        close = ClosePositionAction(exchange=venue, symbol=BTC)
        exit_order = make_market(OrderSide.SELL, 0.1)

        decision = _decide([close, exit_order], _position(1.0))

        assert decision.dropped == (exit_order,)

    def test_a_close_of_a_share_may_close_the_whole_position_so_a_market_order_after_it_is_dropped(
        self, venue, make_market
    ):
        close = ClosePositionAction(exchange=venue, symbol=BTC, closing_ratio=0.5)
        exit_order = make_market(OrderSide.SELL, 0.1)

        decision = _decide([close, exit_order], _position(1.0))

        assert decision.dropped == (exit_order,)

    def test_a_market_order_booked_ahead_of_a_close_of_a_share_is_counted_as_it_is(
        self, venue, make_market
    ):
        exit_order = make_market(OrderSide.SELL, 0.8)
        close = ClosePositionAction(exchange=venue, symbol=BTC, closing_ratio=0.5)

        decision = _decide([exit_order, close], _position(1.0))

        assert decision.actions == (exit_order, close)

    def test_a_close_of_a_share_leaves_a_resting_order_nothing_to_hold(
        self, venue, make_resting
    ):
        close = ClosePositionAction(exchange=venue, symbol=BTC, closing_ratio=0.5)
        resting = make_resting(quantity=0.1, remaining_quantity=0.1)

        decision = _decide([close], _position(1.0), [resting])

        assert decision.cancelled == (resting,)

    def test_a_close_of_a_share_does_not_end_the_exposure(self, venue, make_resting):
        close = ClosePositionAction(exchange=venue, symbol=BTC, closing_ratio=0.5)
        protection = make_resting(
            order_id="sl-1", kind=OrderType.STOP_LOSS, reduce_only=True
        )
        cancel = CancelOrderByIdAction(exchange=venue, symbol=BTC, order_id="sl-1")

        decision = _decide([close, cancel], _position(1.0), [protection])

        assert decision.dropped == (cancel,)

    def test_a_close_resting_at_a_limit_price_is_kept_whatever_the_market_orders_leave(
        self, venue, make_market
    ):
        resting_close = ClosePositionAction(
            exchange=venue, symbol=BTC, limit_price=105.0
        )
        exit_order = make_market(OrderSide.SELL, 1.0)

        decision = _decide([resting_close, exit_order], _position(1.0))

        assert decision.actions == (resting_close, exit_order)

    def test_a_close_of_a_share_resting_at_a_limit_price_drops_a_limit_order_beside_it(
        self, venue, make_limit
    ):
        resting_close = ClosePositionAction(
            exchange=venue, symbol=BTC, closing_ratio=0.6, limit_price=105.0
        )
        exit_limit = make_limit(OrderSide.SELL, 0.1)

        decision = _decide([resting_close, exit_limit], _position(1.0))

        assert decision.dropped == (exit_limit,)

    def test_a_close_of_the_whole_position_resting_at_a_limit_price_drops_a_limit_order_beside_it(
        self, venue, make_limit
    ):
        resting_close = ClosePositionAction(
            exchange=venue, symbol=BTC, limit_price=105.0
        )
        exit_limit = make_limit(OrderSide.SELL, 0.1)

        decision = _decide([resting_close, exit_limit], _position(1.0))

        assert decision.dropped == (exit_limit,)

    def test_a_close_resting_at_a_limit_price_is_counted_after_the_orders_booked_ahead_of_it(
        self, venue, make_limit
    ):
        exit_limit = make_limit(OrderSide.SELL, 0.7)
        resting_close = ClosePositionAction(
            exchange=venue, symbol=BTC, closing_ratio=0.5, limit_price=105.0
        )
        second_limit = make_limit(OrderSide.SELL, 0.2)

        decision = _decide([exit_limit, resting_close, second_limit], _position(1.0))

        assert decision.actions[1] is resting_close
        assert decision.dropped == (second_limit,)

    def test_a_close_resting_at_a_limit_price_does_not_end_the_exposure(
        self, venue, make_resting
    ):
        resting_close = ClosePositionAction(
            exchange=venue, symbol=BTC, limit_price=105.0
        )
        protection = make_resting(
            order_id="sl-1", kind=OrderType.STOP_LOSS, reduce_only=True
        )
        cancel = CancelOrderByIdAction(exchange=venue, symbol=BTC, order_id="sl-1")

        decision = _decide([resting_close, cancel], _position(1.0), [protection])

        assert decision.dropped == (cancel,)


class TestRestingOrdersBooked:
    def test_a_limit_order_closing_within_what_remains_is_kept(self, make_limit):
        exit_limit = make_limit(OrderSide.SELL, 0.5)

        decision = _decide([exit_limit], _position(1.0))

        assert decision.actions == (exit_limit,)

    def test_a_limit_entry_is_dropped(self, make_limit):
        entry = make_limit(OrderSide.BUY, 0.5)

        decision = _decide([entry], _position(1.0))

        assert decision.dropped == (entry,)

    def test_a_limit_order_goes_without_its_protections(self, make_limit):
        exit_limit = make_limit(
            OrderSide.SELL, 0.5, stop_loss=StopLoss(trigger_price=90.0)
        )

        decision = _decide([exit_limit], _position(1.0))

        assert decision.actions[0].stop_loss is None

    def test_a_limit_order_larger_than_what_remains_is_cut(self, make_limit):
        exit_limit = make_limit(OrderSide.SELL, 3.0)

        decision = _decide([exit_limit], _position(1.0))

        assert decision.actions[0].quantity == pytest.approx(1.0)
        assert decision.cut[0].original is exit_limit

    def test_a_trigger_market_order_is_counted_as_a_resting_order(self, make_market):
        trigger = make_market(OrderSide.SELL, 0.5, trigger_price=95.0)
        market = make_market(OrderSide.SELL, 0.8)

        decision = _decide([trigger, market], _position(1.0))

        assert decision.actions[0].quantity == pytest.approx(0.2)
        assert decision.actions[1] is market

    def test_the_cycles_market_orders_are_counted_before_a_limit_order_booked_ahead_of_them(
        self, make_market, make_limit
    ):
        resting = make_limit(OrderSide.SELL, 0.5)
        market = make_market(OrderSide.SELL, 0.8)

        decision = _decide([resting, market], _position(1.0))

        assert decision.actions[0].quantity == pytest.approx(0.2)
        assert decision.actions[1] is market

    def test_a_limit_order_after_the_position_is_used_up_is_dropped(
        self, make_market, make_limit
    ):
        market = make_market(OrderSide.SELL, 1.0)
        exit_limit = make_limit(OrderSide.SELL, 0.5)

        decision = _decide([market, exit_limit], _position(1.0))

        assert decision.dropped == (exit_limit,)

    def test_a_reduce_only_limit_order_is_kept_whatever_it_holds(self, make_limit):
        flagged = make_limit(OrderSide.BUY, 5.0, reduce_only=True)

        decision = _decide([flagged], _position(1.0))

        assert decision.actions == (flagged,)


class TestOrdersAlreadyResting:
    def test_an_order_beyond_what_the_cycles_market_orders_left_is_cancelled(
        self, make_market, make_resting
    ):
        market = make_market(OrderSide.SELL, 0.6)
        resting = make_resting(quantity=0.5, remaining_quantity=0.5)

        decision = _decide([market], _position(1.0), [resting])

        assert decision.cancelled == (resting,)

    def test_an_order_within_what_remains_is_left(self, make_resting):
        resting = make_resting(quantity=0.5, remaining_quantity=0.5)

        decision = _decide([], _position(1.0), [resting])

        assert decision.cancelled == ()

    def test_an_order_on_the_positions_side_is_cancelled(self, make_resting):
        resting = make_resting(side=OrderSide.BUY)

        decision = _decide([], _position(1.0), [resting])

        assert decision.cancelled == (resting,)

    def test_every_order_is_cancelled_when_the_cycle_closes_the_whole_position(
        self, make_market, make_resting
    ):
        close = make_market(OrderSide.SELL, 1.0)
        resting = make_resting(quantity=0.1, remaining_quantity=0.1)

        decision = _decide([close], _position(1.0), [resting])

        assert decision.cancelled == (resting,)

    def test_a_reduce_only_order_is_left(self, make_resting):
        reducing = make_resting(side=OrderSide.BUY, reduce_only=True)

        decision = _decide([], _position(1.0), [reducing])

        assert decision.cancelled == ()

    @pytest.mark.parametrize(
        "kind", [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT, OrderType.TRAILING_STOP]
    )
    def test_a_protection_is_left(self, make_resting, kind):
        protection = make_resting(kind=kind, side=OrderSide.SELL)

        decision = _decide([], None, [protection])

        assert decision.cancelled == ()

    @pytest.mark.parametrize("client_order_id", [OTHER_BOTS_ID, None])
    def test_an_order_another_bot_or_a_hand_placed_is_left(
        self, make_resting, client_order_id
    ):
        foreign = make_resting(side=OrderSide.BUY, client_order_id=client_order_id)

        decision = _decide([], _position(1.0), [foreign])

        assert decision.cancelled == ()

    @pytest.mark.parametrize("kind", [OrderType.LIMIT, OrderType.TRIGGER])
    def test_an_entry_of_this_bot_is_cancelled_on_a_symbol_holding_no_position(
        self, make_resting, kind
    ):
        entry = make_resting(kind=kind)

        decision = _decide([], None, [entry])

        assert decision.cancelled == (entry,)

    def test_an_entry_the_venue_states_no_reduce_only_flag_for_is_judged_by_side_and_size(
        self, make_resting
    ):
        unflagged = make_resting(reduce_only=None, side=OrderSide.BUY)
        fitting = make_resting(
            order_id="fitting", reduce_only=None, quantity=0.5, remaining_quantity=0.5
        )

        decision = _decide([], _position(1.0), [unflagged, fitting])

        assert decision.cancelled == (unflagged,)

    @pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
    def test_every_entry_is_cancelled_on_a_symbol_holding_no_position_whatever_its_side(
        self, make_resting, side
    ):
        entry = make_resting(side=side)

        decision = _decide([], None, [entry])

        assert decision.cancelled == (entry,)

    def test_a_trigger_with_no_id_is_cancelled_where_the_venue_keeps_none(
        self, make_resting
    ):
        trigger = make_resting(
            kind=OrderType.TRIGGER, side=OrderSide.BUY, client_order_id=None
        )

        decision = _decide([], _position(1.0), [trigger], labels_resting_triggers=False)

        assert decision.cancelled == (trigger,)

    def test_a_trigger_with_no_id_is_left_where_the_venue_keeps_it(self, make_resting):
        trigger = make_resting(
            kind=OrderType.TRIGGER, side=OrderSide.BUY, client_order_id=None
        )

        decision = _decide([], _position(1.0), [trigger])

        assert decision.cancelled == ()

    def test_a_limit_order_with_no_id_is_left_where_the_venue_keeps_no_trigger_id(
        self, make_resting
    ):
        limit = make_resting(
            kind=OrderType.LIMIT, side=OrderSide.BUY, client_order_id=None
        )

        decision = _decide([], _position(1.0), [limit], labels_resting_triggers=False)

        assert decision.cancelled == ()

    def test_an_order_whose_side_the_venue_does_not_state_is_cancelled(
        self, make_resting
    ):
        sideless = make_resting(side=None)

        decision = _decide([], _position(1.0), [sideless])

        assert decision.cancelled == (sideless,)

    def test_an_order_whose_size_the_venue_does_not_state_is_cancelled(
        self, make_resting
    ):
        sizeless = make_resting(quantity=None, remaining_quantity=None)

        decision = _decide([], _position(1.0), [sizeless])

        assert decision.cancelled == (sizeless,)

    def test_an_order_equal_to_the_position_is_left(self, make_resting):
        whole = make_resting(quantity=1.0, remaining_quantity=1.0)

        decision = _decide([], _position(1.0), [whole])

        assert decision.cancelled == ()

    def test_an_order_larger_than_the_position_is_cancelled(self, make_resting):
        too_large = make_resting(quantity=1.5, remaining_quantity=1.5)

        decision = _decide([], _position(1.0), [too_large])

        assert decision.cancelled == (too_large,)

    def test_a_smaller_order_after_a_cancelled_one_is_left(self, make_resting):
        too_large = make_resting(order_id="large", quantity=1.5, remaining_quantity=1.5)
        small = make_resting(order_id="small", quantity=0.5, remaining_quantity=0.5)

        decision = _decide([], _position(1.0), [too_large, small])

        assert decision.cancelled == (too_large,)

    def test_the_orders_cancelled_come_in_the_sequence_the_venue_listed_them(
        self, make_resting
    ):
        first = make_resting(order_id="first", side=OrderSide.BUY)
        second = make_resting(order_id="second", side=OrderSide.BUY)

        decision = _decide([], _position(1.0), [second, first])

        assert decision.cancelled == (second, first)

    def test_the_newer_of_two_orders_exceeding_the_position_together_is_cancelled(
        self, make_resting
    ):
        older = make_resting(order_id="older", quantity=0.6, remaining_quantity=0.6)
        newer = make_resting(order_id="newer", quantity=0.6, remaining_quantity=0.6)

        decision = _decide([], _position(1.0), [older, newer])

        assert decision.cancelled == (newer,)

    def test_an_order_the_strategy_cancels_counts_for_nothing(
        self, venue, make_resting, make_limit
    ):
        resting = make_resting(order_id="rung-1", quantity=0.6, remaining_quantity=0.6)
        cancel = CancelOrderByIdAction(exchange=venue, symbol=BTC, order_id="rung-1")
        exit_limit = make_limit(OrderSide.SELL, 0.9)

        decision = _decide([cancel, exit_limit], _position(1.0), [resting])

        assert decision.cancelled == ()
        assert decision.actions[1].quantity == 0.9

    def test_an_order_a_booked_one_makes_no_room_for_is_not_kept_for_it(
        self, make_resting, make_limit
    ):
        resting = make_resting(quantity=0.6, remaining_quantity=0.6)
        exit_limit = make_limit(OrderSide.SELL, 0.6)

        decision = _decide([exit_limit], _position(1.0), [resting])

        assert decision.actions[0].quantity == pytest.approx(0.4)
        assert decision.cancelled == ()


class TestCancels:
    def test_a_cancel_by_id_of_a_resting_entry_is_kept(self, venue, make_resting):
        resting = make_resting(order_id="rung-1")
        cancel = CancelOrderByIdAction(exchange=venue, symbol=BTC, order_id="rung-1")

        decision = _decide([cancel], _position(1.0), [resting])

        assert decision.actions == (cancel,)

    @pytest.mark.parametrize(
        "kind", [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT, OrderType.TRAILING_STOP]
    )
    def test_a_cancel_by_id_of_a_protection_is_dropped_while_the_position_stands(
        self, venue, make_resting, kind
    ):
        protection = make_resting(order_id="sl-1", kind=kind, reduce_only=True)
        cancel = CancelOrderByIdAction(exchange=venue, symbol=BTC, order_id="sl-1")

        decision = _decide([cancel], _position(1.0), [protection])

        assert decision.dropped == (cancel,)

    def test_a_cancel_by_id_of_a_protection_is_kept_on_a_symbol_holding_no_position(
        self, venue, make_resting
    ):
        protection = make_resting(
            order_id="sl-1", kind=OrderType.STOP_LOSS, reduce_only=True
        )
        cancel = CancelOrderByIdAction(exchange=venue, symbol=BTC, order_id="sl-1")

        decision = _decide([cancel], None, [protection])

        assert decision.actions == (cancel,)

    def test_a_cancel_by_id_of_a_protection_is_kept_when_the_cycle_closes_the_position(
        self, venue, make_market, make_resting
    ):
        protection = make_resting(
            order_id="sl-1", kind=OrderType.STOP_LOSS, reduce_only=True
        )
        exit_order = make_market(OrderSide.SELL, 1.0)
        cancel = CancelOrderByIdAction(exchange=venue, symbol=BTC, order_id="sl-1")

        decision = _decide([exit_order, cancel], _position(1.0), [protection])

        assert decision.actions == (exit_order, cancel)

    def test_a_cancel_by_id_booked_before_the_closing_order_is_kept_all_the_same(
        self, venue, make_market, make_resting
    ):
        protection = make_resting(
            order_id="sl-1", kind=OrderType.STOP_LOSS, reduce_only=True
        )
        cancel = CancelOrderByIdAction(exchange=venue, symbol=BTC, order_id="sl-1")
        exit_order = make_market(OrderSide.SELL, 1.0)

        decision = _decide([cancel, exit_order], _position(1.0), [protection])

        assert decision.actions == (cancel, exit_order)

    def test_a_cancel_by_id_of_a_protection_is_dropped_when_the_cycle_closes_part_of_the_position(
        self, venue, make_market, make_resting
    ):
        protection = make_resting(
            order_id="sl-1", kind=OrderType.STOP_LOSS, reduce_only=True
        )
        exit_order = make_market(OrderSide.SELL, 0.5)
        cancel = CancelOrderByIdAction(exchange=venue, symbol=BTC, order_id="sl-1")

        decision = _decide([exit_order, cancel], _position(1.0), [protection])

        assert decision.dropped == (cancel,)

    def test_a_cancel_by_id_of_an_order_the_snapshot_does_not_hold_is_judged_as_a_protection(
        self, venue
    ):
        cancel = CancelOrderByIdAction(exchange=venue, symbol=BTC, order_id="unseen")

        decision = _decide([cancel], _position(1.0), [])

        assert decision.dropped == (cancel,)

    def test_a_cancel_by_a_tag_of_the_strategy_is_kept(self, venue):
        cancel = CancelOrdersForSymbolAction(
            exchange=venue, symbol=BTC, order_tag="1h-rung"
        )

        decision = _decide([cancel], _position(1.0))

        assert decision.actions == (cancel,)

    @pytest.mark.parametrize("tag", ["sl", "tp"])
    def test_a_cancel_by_a_protections_tag_is_dropped_while_the_position_stands(
        self, venue, tag
    ):
        cancel = CancelOrdersForSymbolAction(exchange=venue, symbol=BTC, order_tag=tag)

        decision = _decide([cancel], _position(1.0))

        assert decision.dropped == (cancel,)

    def test_a_cancel_by_a_protections_tag_is_kept_when_the_cycle_closes_the_position(
        self, venue, make_market
    ):
        exit_order = make_market(OrderSide.SELL, 1.0)
        cancel = CancelOrdersForSymbolAction(exchange=venue, symbol=BTC, order_tag="sl")

        decision = _decide([exit_order, cancel], _position(1.0))

        assert decision.actions == (exit_order, cancel)

    def test_a_cancel_of_every_order_is_dropped_while_the_symbol_holds_a_position(
        self, venue
    ):
        cancel = CancelOrdersForSymbolAction(exchange=venue, symbol=BTC)

        decision = _decide([cancel], _position(1.0))

        assert decision.dropped == (cancel,)

    def test_a_cancel_of_every_order_is_dropped_though_the_cycle_closes_the_position(
        self, venue, make_market
    ):
        exit_order = make_market(OrderSide.SELL, 1.0)
        cancel = CancelOrdersForSymbolAction(exchange=venue, symbol=BTC)

        decision = _decide([exit_order, cancel], _position(1.0))

        assert decision.dropped == (cancel,)

    def test_a_cancel_of_every_order_is_kept_on_a_symbol_holding_no_position(
        self, venue
    ):
        cancel = CancelOrdersForSymbolAction(exchange=venue, symbol=BTC)

        decision = _decide([cancel], None)

        assert decision.actions == (cancel,)

    def test_a_cancel_of_every_order_leaves_no_resting_order_to_cancel_again(
        self, venue, make_resting
    ):
        cancel = CancelOrdersForSymbolAction(exchange=venue, symbol=BTC)
        resting = make_resting()

        decision = _decide([cancel], None, [resting])

        assert decision.cancelled == ()

    def test_a_cancel_by_tag_leaves_no_order_of_the_tag_to_cancel_again(
        self, venue, make_resting
    ):
        cancel = CancelOrdersForSymbolAction(
            exchange=venue, symbol=BTC, order_tag="1h-fast"
        )
        resting = make_resting(side=OrderSide.BUY)

        decision = _decide([cancel], _position(1.0), [resting])

        assert decision.cancelled == ()


class TestModifications:
    def _modification(
        self, venue, make_limit, order_id="rung-1", quantity=0.5, **fields
    ) -> FuturesOrderModifyAction:
        replacement = make_limit(OrderSide.SELL, quantity, price=106.0, **fields)
        return FuturesOrderModifyAction(
            exchange=venue,
            symbol=BTC,
            modifications=[OrderReplacement(order_id=order_id, order=replacement)],
        )

    def test_a_price_move_of_an_order_on_the_closing_side_is_kept(
        self, venue, make_limit, make_resting
    ):
        resting = make_resting(order_id="rung-1", quantity=0.5)
        modify = self._modification(venue, make_limit)

        decision = _decide([modify], _position(1.0), [resting])

        assert decision.actions == (modify,)

    def test_a_modification_raising_the_quantity_is_dropped(
        self, venue, make_limit, make_resting
    ):
        resting = make_resting(order_id="rung-1", quantity=0.5)
        modify = self._modification(venue, make_limit, quantity=0.6)

        decision = _decide([modify], _position(1.0), [resting])

        assert decision.actions == ()
        assert decision.dropped == (modify.modifications[0].order,)

    def test_a_modification_lowering_the_quantity_is_kept(
        self, venue, make_limit, make_resting
    ):
        resting = make_resting(order_id="rung-1", quantity=0.5)
        modify = self._modification(venue, make_limit, quantity=0.4)

        decision = _decide([modify], _position(1.0), [resting])

        assert decision.actions == (modify,)

    def test_a_modification_of_an_entry_on_the_positions_side_is_dropped(
        self, venue, make_limit, make_resting
    ):
        resting = make_resting(order_id="rung-1", side=OrderSide.BUY)
        modify = self._modification(venue, make_limit)

        decision = _decide([modify], _position(1.0), [resting])

        assert decision.actions == ()

    def test_a_modification_of_an_order_the_snapshot_does_not_hold_is_dropped(
        self, venue, make_limit
    ):
        modify = self._modification(venue, make_limit, order_id="unseen")

        decision = _decide([modify], _position(1.0), [])

        assert decision.actions == ()

    def test_a_modification_of_an_order_the_cycle_cancels_is_dropped(
        self, venue, make_limit, make_market, make_resting
    ):
        resting = make_resting(order_id="rung-1", quantity=0.5)
        exit_order = make_market(OrderSide.SELL, 0.8)
        modify = self._modification(venue, make_limit)

        decision = _decide([exit_order, modify], _position(1.0), [resting])

        assert decision.cancelled == (resting,)
        assert decision.actions == (exit_order,)

    def test_a_modification_on_a_symbol_holding_no_position_is_dropped(
        self, venue, make_limit, make_resting
    ):
        resting = make_resting(order_id="rung-1")
        modify = self._modification(venue, make_limit)

        decision = _decide([modify], None, [resting])

        assert decision.actions == ()

    def test_the_replacement_of_an_order_not_reduce_only_goes_without_its_protections(
        self, venue, make_limit, make_resting
    ):
        resting = make_resting(order_id="rung-1", quantity=0.5)
        modify = self._modification(
            venue, make_limit, stop_loss=StopLoss(trigger_price=90.0)
        )

        decision = _decide([modify], _position(1.0), [resting])

        sent = decision.actions[0]
        assert sent.modifications[0].order.stop_loss is None
        assert sent.modifications[0].order_id == "rung-1"
        assert modify.modifications[0].order.stop_loss is not None

    def test_a_modification_of_a_reduce_only_order_keeps_its_replacement_as_booked(
        self, venue, make_limit, make_resting
    ):
        resting = make_resting(order_id="rung-1", quantity=0.5, reduce_only=True)
        modify = self._modification(
            venue, make_limit, stop_loss=StopLoss(trigger_price=90.0)
        )

        decision = _decide([modify], _position(1.0), [resting])

        assert decision.actions == (modify,)

    def test_a_modification_spanning_two_symbols_keeps_the_other_symbols_member(
        self, venue, make_limit, make_resting
    ):
        resting = make_resting(order_id="rung-1", quantity=0.5)
        other = make_limit(OrderSide.BUY, 1.0, symbol=ETH)
        ours = make_limit(OrderSide.SELL, 0.6, price=106.0)
        modify = FuturesOrderModifyAction(
            exchange=venue,
            symbol=ETH,
            modifications=[
                OrderReplacement(order_id="eth-rung", order=other),
                OrderReplacement(order_id="rung-1", order=ours),
            ],
        )

        decision = _decide([modify], _position(1.0), [resting])

        sent = decision.actions[0]
        assert [m.order for m in sent.modifications] == [other]
        assert sent.symbol == ETH
        assert sent.id == modify.id


class TestBatches:
    def test_a_batch_of_orders_all_let_through_is_returned_as_booked(
        self, venue, make_limit
    ):
        first = make_limit(OrderSide.SELL, 0.3)
        second = make_limit(OrderSide.SELL, 0.3)
        batch = FuturesOrderBatchAction(
            exchange=venue, symbol=BTC, orders=[first, second]
        )

        decision = _decide([batch], _position(1.0))

        assert decision.actions == (batch,)
        assert decision.actions[0] is batch

    def test_a_batch_keeps_the_members_exit_only_mode_lets_through(
        self, venue, make_limit
    ):
        entry = make_limit(OrderSide.BUY, 0.3)
        exit_limit = make_limit(OrderSide.SELL, 0.3)
        batch = FuturesOrderBatchAction(
            exchange=venue, symbol=BTC, orders=[entry, exit_limit]
        )

        decision = _decide([batch], _position(1.0))

        sent = decision.actions[0]
        assert sent.orders == [exit_limit]
        assert sent.id == batch.id
        assert decision.dropped == (entry,)
        assert batch.orders == [entry, exit_limit]

    def test_a_batch_whose_members_are_all_dropped_is_not_sent(self, venue, make_limit):
        batch = FuturesOrderBatchAction(
            exchange=venue,
            symbol=BTC,
            orders=[make_limit(OrderSide.BUY, 0.3), make_limit(OrderSide.BUY, 0.3)],
        )

        decision = _decide([batch], _position(1.0))

        assert decision.actions == ()

    def test_a_batch_spanning_two_symbols_sends_the_other_symbols_orders_whole(
        self, venue, make_limit
    ):
        entry = make_limit(OrderSide.BUY, 0.3)
        other_entry = make_limit(OrderSide.BUY, 5.0, symbol=ETH)
        batch = FuturesOrderBatchAction(
            exchange=venue, symbol=BTC, orders=[entry, other_entry]
        )

        decision = _decide([batch], _position(1.0))

        sent = decision.actions[0]
        assert sent.orders == [other_entry]
        assert sent.symbol == ETH

    def test_the_members_of_a_batch_are_counted_in_booking_order(
        self, venue, make_market
    ):
        first = make_market(OrderSide.SELL, 0.7)
        second = make_market(OrderSide.SELL, 0.7)
        batch = FuturesOrderBatchAction(
            exchange=venue, symbol=BTC, orders=[first, second]
        )

        decision = _decide([batch], _position(1.0))

        sent = decision.actions[0]
        assert sent.orders[0] is first
        assert sent.orders[1].quantity == pytest.approx(0.3)

    def test_a_cut_member_is_counted_with_the_standalone_orders_of_the_symbol(
        self, venue, make_market
    ):
        standalone = make_market(OrderSide.SELL, 0.7)
        member = make_market(OrderSide.SELL, 0.7)
        batch = FuturesOrderBatchAction(exchange=venue, symbol=BTC, orders=[member])

        decision = _decide([standalone, batch], _position(1.0))

        assert decision.actions[1].orders[0].quantity == pytest.approx(0.3)


def test_the_quantity_cut_is_a_whole_number_of_steps(make_market):
    exit_order = make_market(OrderSide.SELL, 3.0)

    decision = _decide([exit_order], _position(0.123456789))

    assert to_step_count(decision.actions[0].quantity) == to_step_count(0.123456789)
