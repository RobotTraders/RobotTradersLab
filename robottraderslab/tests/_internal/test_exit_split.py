import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab._internal import split_exits
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ExchangeTransientError,
)
from robottraderslab.exchanges import Balance, OrderSide, PositionSide, PositionSnapshot
from robottraderslab.futures import FuturesAccount, FuturesOrderBuilder
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    BookKeeper,
    StrategyRequirements,
    TrackedPosition,
    TrackingId,
    profile_tag,
)
from robottraderslab_core.engine import execute_trading_actions
from robottraderslab_core.exchanges import BaseExchangeAction, PlacedOrder
from robottraderslab_core.futures import BatchableOrderAction, ClosePositionAction

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")
ALPHA = TrackingId("BTC/USDT:USDT@1d#alpha")


@pytest.fixture
def venue(make_venue: Callable[..., Mock]) -> Mock:
    return make_venue(quantity_step=0.5)


@pytest.fixture
async def futures_account(venue: Mock) -> FuturesAccount:
    return await _read_account(venue, [BTC, ETH])


@pytest.fixture
def fine_venue(make_venue: Callable[..., Mock]) -> Mock:
    return make_venue()


@pytest.fixture
async def fine_account(fine_venue: Mock) -> FuturesAccount:
    return await _read_account(fine_venue, [BTC, ETH])


@pytest.fixture
def failing_read(fine_venue: Mock) -> Mock:
    fine_venue.get_open_positions.side_effect = ExchangeRecoverableError("down")
    return fine_venue


@pytest.fixture
async def strict_account(make_venue: Callable[..., Mock]) -> FuturesAccount:
    venue = make_venue(quantity_step=0.5, minimum_order_value=1000.0)
    return await _read_account(venue, [BTC])


class TestAnExitWithinThePosition:
    async def test_it_goes_reduce_only(self, futures_account):
        bookkeeper = _booked_exit(futures_account, PositionSide.LONG, 1.0)

        await _split_holding(futures_account, bookkeeper)

        assert _sent(bookkeeper) == [(OrderSide.SELL, 1.0, True)]

    async def test_it_keeps_its_id_so_the_client_order_id_it_was_booked_under_stays(
        self, futures_account
    ):
        exit_order = _tracked_exit(futures_account, PositionSide.LONG, 1.0)
        bookkeeper = _bookkeeper(exit_order)

        await _split_holding(futures_account, bookkeeper)

        assert bookkeeper.list_actions()[0].id == exit_order.id

    async def test_a_short_is_closed_by_a_reduce_only_buy(self, futures_account):
        bookkeeper = _booked_exit(futures_account, PositionSide.SHORT, 1.0)

        await _split_holding(futures_account, bookkeeper, PositionSide.SHORT, 1.0)

        assert _sent(bookkeeper) == [(OrderSide.BUY, 1.0, True)]

    async def test_a_limit_exit_rests_reduce_only(self, futures_account):
        resting = _resting_exit(futures_account, 1.0, "a")
        bookkeeper = _bookkeeper(resting)

        await _split_holding(futures_account, bookkeeper)

        assert _sent(bookkeeper) == [(OrderSide.SELL, 1.0, True)]

    async def test_an_exit_booked_reduce_only_goes_as_booked(self, futures_account):
        exit_order = futures_account.long_exit(BTC, 1.0).build()
        bookkeeper = _bookkeeper(exit_order)

        await _split_holding(futures_account, bookkeeper)

        assert bookkeeper.list_actions() == [exit_order]

    async def test_nothing_changes_when_nothing_booked_is_an_exit(
        self, futures_account
    ):
        entry = futures_account.long_entry(BTC, 1.0).build()
        bookkeeper = _bookkeeper(entry)

        await _split_holding(futures_account, bookkeeper)

        assert bookkeeper.list_actions() == [entry]


class TestAnExitOverThePosition:
    async def test_the_surplus_goes_as_a_plain_order_after_the_reduce_only_part(
        self, futures_account
    ):
        bookkeeper = _booked_exit(futures_account, PositionSide.LONG, 3.0)

        await _split_holding(futures_account, bookkeeper)

        assert _sent(bookkeeper) == [
            (OrderSide.SELL, 2.0, True),
            (OrderSide.SELL, 1.0, False),
        ]

    async def test_the_remainder_is_the_same_exit_under_a_fresh_id(
        self, futures_account
    ):
        bookkeeper = _booked_exit(futures_account, PositionSide.LONG, 3.0)

        await _split_holding(futures_account, bookkeeper)

        reducing, remainder = bookkeeper.list_actions()
        assert remainder.id != reducing.id
        assert remainder.tag == "alpha"
        assert remainder.reason == "exit"
        assert remainder.is_exit is True

    async def test_the_remainder_carries_no_protection_and_no_callback(
        self, futures_account
    ):
        exit_order = (
            _tracked_builder(futures_account, PositionSide.LONG, 3.0)
            .stop_loss(90.0)
            .when_filled(_ignore_fill)
            .build()
        )
        bookkeeper = _bookkeeper(exit_order)

        await _split_holding(futures_account, bookkeeper)

        reducing, remainder = bookkeeper.list_actions()
        assert reducing.stop_loss is not None
        assert reducing.on_filled is not None
        assert remainder.stop_loss is None
        assert remainder.on_filled is None

    async def test_an_action_booked_after_the_exit_waits_for_both_parts(
        self, futures_account
    ):
        exit_order = _tracked_exit(futures_account, PositionSide.LONG, 3.0)
        bookkeeper = _bookkeeper(exit_order)
        entry = futures_account.short_entry(BTC, 1.0).build()
        bookkeeper.add(entry, after=exit_order)

        await _split_holding(futures_account, bookkeeper)

        _, remainder, _ = bookkeeper.list_actions()
        assert bookkeeper.declared_waits()[entry.id] == (exit_order.id, remainder.id)

    async def test_a_resting_exit_splits_after_the_exits_sent_at_once(
        self, futures_account
    ):
        resting = _resting_exit(futures_account, 3.0, "a")
        at_once = _tracked_exit(futures_account, PositionSide.LONG, 1.0, tag="b")
        bookkeeper = _bookkeeper(resting, at_once)

        await _split_holding(futures_account, bookkeeper)

        assert _sent(bookkeeper) == [
            (OrderSide.SELL, 1.0, True),
            (OrderSide.SELL, 2.0, False),
            (OrderSide.SELL, 1.0, True),
        ]

    async def test_resting_exits_share_the_room_the_position_leaves(
        self, futures_account
    ):
        first, second = (_resting_exit(futures_account, 1.5, tag) for tag in ("a", "b"))
        bookkeeper = _bookkeeper(first, second)

        await _split_holding(futures_account, bookkeeper)

        assert _sent(bookkeeper) == [
            (OrderSide.SELL, 1.5, True),
            (OrderSide.SELL, 0.5, True),
            (OrderSide.SELL, 1.0, False),
        ]

    async def test_a_resting_close_keeps_its_share_of_the_room(
        self, futures_account, venue
    ):
        close = ClosePositionAction(
            exchange=venue, symbol=BTC, closing_ratio=0.5, limit_price=120.0
        )
        resting = _resting_exit(futures_account, 2.0, "a")
        bookkeeper = _bookkeeper(close, resting)

        await _split_holding(futures_account, bookkeeper)

        assert _sent_without_closes(bookkeeper) == [
            (OrderSide.SELL, 1.0, True),
            (OrderSide.SELL, 1.0, False),
        ]

    async def test_a_close_of_a_share_leaves_the_rest_for_the_exits_after_it(
        self, futures_account, venue
    ):
        close = ClosePositionAction(exchange=venue, symbol=BTC, closing_ratio=0.5)
        exit_order = _tracked_exit(futures_account, PositionSide.LONG, 2.0)
        bookkeeper = _bookkeeper(close, exit_order)

        await _split_holding(futures_account, bookkeeper)

        assert _sent_without_closes(bookkeeper) == [
            (OrderSide.SELL, 1.0, True),
            (OrderSide.SELL, 1.0, False),
        ]

    async def test_a_resting_close_of_a_short_keeps_its_share_of_the_room(
        self, futures_account, venue
    ):
        close = ClosePositionAction(
            exchange=venue, symbol=BTC, closing_ratio=0.5, limit_price=80.0
        )
        resting = (
            futures_account.short_exit(BTC, 2.0, reduce_only=False)
            .limit(90.0)
            .tag("a")
            .build()
        )
        bookkeeper = _bookkeeper(close, resting)

        await _split_holding(futures_account, bookkeeper, PositionSide.SHORT, 2.0)

        assert _sent_without_closes(bookkeeper) == [
            (OrderSide.BUY, 1.0, True),
            (OrderSide.BUY, 1.0, False),
        ]

    async def test_a_close_of_a_share_of_a_short_leaves_the_rest_for_the_exits_after_it(
        self, futures_account, venue
    ):
        close = ClosePositionAction(exchange=venue, symbol=BTC, closing_ratio=0.5)
        exit_order = _tracked_exit(futures_account, PositionSide.SHORT, 2.0)
        bookkeeper = _bookkeeper(close, exit_order)

        await _split_holding(futures_account, bookkeeper, PositionSide.SHORT, 2.0)

        assert _sent_without_closes(bookkeeper) == [
            (OrderSide.BUY, 1.0, True),
            (OrderSide.BUY, 1.0, False),
        ]

    async def test_a_member_of_a_batch_goes_reduce_only_and_its_remainder_follows_the_batch(
        self, futures_account
    ):
        batch = _batch_exit(futures_account)
        bookkeeper = _bookkeeper(batch, after=[])

        await _split_holding(futures_account, bookkeeper)

        batched, remainder = bookkeeper.list_actions()
        assert [(o.quantity, o.reduce_only) for o in batched.orders] == [(2.0, True)]
        assert (remainder.quantity, remainder.reduce_only) == (1.0, False)
        assert bookkeeper.declared_waits()[remainder.id] == (batch.id,)

    async def test_a_remainder_under_the_minimum_order_value_is_denied(
        self, strict_account
    ):
        exit_order = (
            strict_account.long_exit(BTC, 3.0, reduce_only=False)
            .limit(400.0)
            .tag("a")
            .build()
        )
        bookkeeper = _bookkeeper(exit_order)

        await _split_holding(strict_account, bookkeeper)

        reducing, remainder = bookkeeper.list_actions()
        assert reducing.denial is None
        assert "minimum order value" in remainder.denial


class TestTwoProfilesOnOppositeSidesOfOneSymbol:
    async def test_each_exit_takes_its_share_of_the_net_position_in_booking_order(
        self, futures_account
    ):
        first = _tracked_exit(futures_account, PositionSide.LONG, 1.0, tag="a")
        second = _tracked_exit(futures_account, PositionSide.LONG, 2.0, tag="c")
        bookkeeper = _bookkeeper(first, second)

        await _split_holding(futures_account, bookkeeper)

        assert _sent(bookkeeper) == [
            (OrderSide.SELL, 1.0, True),
            (OrderSide.SELL, 1.0, True),
            (OrderSide.SELL, 1.0, False),
        ]

    async def test_an_exit_on_the_side_of_the_net_position_stays_plain(
        self, futures_account
    ):
        exit_order = _tracked_exit(futures_account, PositionSide.LONG, 1.0)
        bookkeeper = _bookkeeper(exit_order)

        await _split_holding(futures_account, bookkeeper, PositionSide.SHORT, 2.0)

        assert bookkeeper.list_actions() == [exit_order]
        assert exit_order.reduce_only is False

    async def test_the_exit_that_follows_a_plain_one_closes_the_position_it_opened(
        self, futures_account
    ):
        sell = _tracked_exit(futures_account, PositionSide.LONG, 1.0, tag="a")
        buy = _tracked_exit(futures_account, PositionSide.SHORT, 1.0, tag="b")
        bookkeeper = _bookkeeper(sell, buy)

        await _split_flat(futures_account, bookkeeper)

        assert _sent(bookkeeper) == [
            (OrderSide.SELL, 1.0, False),
            (OrderSide.BUY, 1.0, True),
        ]


class TestAnExitBookedReduceOnly:
    async def test_the_surplus_is_cut_and_warned_about(self, futures_account, caplog):
        bookkeeper = _booked_reduce_only_exit(futures_account, 3.0, "a")

        await _logging_warnings(caplog, _split_holding(futures_account, bookkeeper))

        assert _sent(bookkeeper) == [(OrderSide.SELL, 2.0, True)]
        assert str(BTC) in caplog.text
        assert "profile a" in caplog.text
        assert "cut by 1" in caplog.text

    async def test_an_exit_without_a_tag_is_cut_with_no_profile_named(
        self, futures_account, caplog
    ):
        bookkeeper = _bookkeeper(futures_account.long_exit(BTC, 3.0))

        await _logging_warnings(caplog, _split_holding(futures_account, bookkeeper))

        assert "the exit of no profile is cut by 1" in caplog.text

    async def test_the_net_moves_only_by_what_is_sent(self, futures_account):
        long_exit = _reduce_only_exit(futures_account, 3.0, "a")
        short_exit = futures_account.short_exit(BTC, 1.0).tag("b").build()
        bookkeeper = _bookkeeper(long_exit, short_exit)

        await _split_holding(futures_account, bookkeeper)

        assert _sent(bookkeeper) == [(OrderSide.SELL, 2.0, True)]

    async def test_an_exit_on_a_flat_symbol_is_cut_whole(self, futures_account, caplog):
        bookkeeper = _booked_reduce_only_exit(futures_account, 1.0, "a")

        await _logging_warnings(caplog, _split_flat(futures_account, bookkeeper))

        assert bookkeeper.list_actions() == []
        assert "cut by 1" in caplog.text

    async def test_a_target_that_only_shrinks_is_cut_to_the_position_the_venue_holds(
        self, futures_account
    ):
        target_builder = futures_account.long_target(
            BTC, _target_snapshot(futures_account)
        ).quantity(0.0, 100.0)
        bookkeeper = _bookkeeper(target_builder)

        await _split_holding(futures_account, bookkeeper, PositionSide.LONG, 1.0)

        assert _sent(bookkeeper) == [(OrderSide.SELL, 1.0, True)]


class TestAnExitThatReducesNothing:
    async def test_a_priced_plain_exit_under_the_minimum_order_value_is_denied(
        self, strict_account
    ):
        exit_order = (
            strict_account.long_exit(BTC, 1.0, reduce_only=False)
            .limit(400.0)
            .tag("a")
            .build()
        )
        bookkeeper = _bookkeeper(exit_order)

        await _split_holding(strict_account, bookkeeper, PositionSide.SHORT)

        (denied,) = bookkeeper.list_actions()
        assert denied.id == exit_order.id
        assert "minimum order value" in denied.denial

    async def test_a_trigger_exit_that_reduces_nothing_is_measured_at_its_trigger_price(
        self, strict_account
    ):
        exit_order = (
            strict_account.long_exit(BTC, 1.0, reduce_only=False)
            .trigger(400.0)
            .tag("a")
            .build()
        )
        bookkeeper = _bookkeeper(exit_order)

        await _split_holding(strict_account, bookkeeper, PositionSide.SHORT)

        (denied,) = bookkeeper.list_actions()
        assert "minimum order value" in denied.denial


class TestAnExitThatReplacesARestingOrder:
    async def test_it_is_cut_to_the_position_and_sends_no_remainder(
        self, futures_account, caplog
    ):
        replacement = _resting_exit_builder(futures_account, 3.0)
        bookkeeper = _booking_replacement(futures_account, replacement)

        await _logging_warnings(caplog, _split_holding(futures_account, bookkeeper))

        (modification,) = bookkeeper.list_actions()
        (replaced,) = modification.modifications
        assert (replaced.order.quantity, replaced.order.reduce_only) == (2.0, True)
        assert "cut by 1" in caplog.text

    async def test_it_is_left_out_when_the_position_holds_nothing_to_reduce(
        self, futures_account
    ):
        replacement = _resting_exit_builder(futures_account, 1.0)
        bookkeeper = _booking_replacement(futures_account, replacement)

        await _split_flat(futures_account, bookkeeper)

        assert bookkeeper.list_actions() == []


class TestAnExitOnASymbolTheTrackerCannotAnchor:
    async def test_the_surplus_is_cut_instead_of_sent_plain(
        self, futures_account, caplog
    ):
        strategy_requirements = _declared(futures_account)
        account_snapshots = _holding(futures_account, PositionSide.LONG, 2.0)
        _track_alpha(strategy_requirements, futures_account, account_snapshots)
        bookkeeper = _booked_exit(
            futures_account, PositionSide.LONG, 3.0, tag="1d-alpha"
        )

        await _logging_warnings(
            caplog, split_exits(strategy_requirements, account_snapshots, bookkeeper)
        )

        assert _sent(bookkeeper) == [(OrderSide.SELL, 2.0, True)]
        assert "profile 1d-alpha" in caplog.text
        assert "cut by 1" in caplog.text

    async def test_a_symbol_the_tracker_anchors_sends_the_remainder(
        self, futures_account
    ):
        strategy_requirements = _declared(futures_account)
        account_snapshots = _account_snapshots(futures_account, {})
        _track_alpha(strategy_requirements, futures_account, account_snapshots)
        bookkeeper = _booked_exit(futures_account, PositionSide.LONG, 3.0)

        await split_exits(strategy_requirements, account_snapshots, bookkeeper)

        assert _sent(bookkeeper) == [(OrderSide.SELL, 3.0, False)]


class TestWhatMovesThePositionBeforeAnExit:
    async def test_an_entry_in_the_direction_of_the_net_raises_what_the_exit_may_reduce(
        self, futures_account
    ):
        entry = futures_account.long_entry(BTC, 1.0).build()
        exit_order = _reduce_only_exit(futures_account, 3.0, "a")
        bookkeeper = _bookkeeper(entry, exit_order)

        await _split_holding(futures_account, bookkeeper)

        assert _sent(bookkeeper) == [
            (OrderSide.BUY, 1.0, False),
            (OrderSide.SELL, 3.0, True),
        ]

    async def test_an_entry_against_the_net_leaves_the_exit_judged_against_the_net_held(
        self, futures_account
    ):
        entry = futures_account.short_entry(BTC, 1.0).build()
        exit_order = _reduce_only_exit(futures_account, 2.0, "a")
        bookkeeper = _bookkeeper(entry, exit_order)

        await _split_holding(futures_account, bookkeeper)

        assert _sent(bookkeeper) == [
            (OrderSide.SELL, 1.0, False),
            (OrderSide.SELL, 2.0, True),
        ]

    async def test_an_entry_the_venue_may_refuse_does_not_take_the_exit_off_the_position(
        self, futures_account
    ):
        entry = futures_account.long_entry(BTC, 1.0).build()
        exit_order = (
            futures_account.short_exit(BTC, 1.0, reduce_only=False).tag("b").build()
        )
        bookkeeper = _bookkeeper(entry, exit_order)

        await _split_holding(futures_account, bookkeeper, PositionSide.SHORT, 1.0)

        assert _sent(bookkeeper) == [
            (OrderSide.BUY, 1.0, False),
            (OrderSide.BUY, 1.0, True),
        ]

    async def test_a_close_of_the_symbol_booked_earlier_leaves_nothing_to_reduce(
        self, futures_account, venue
    ):
        close = ClosePositionAction(exchange=venue, symbol=BTC)
        exit_order = _tracked_exit(futures_account, PositionSide.LONG, 1.0)
        bookkeeper = _bookkeeper(close, exit_order)

        await _split_holding(futures_account, bookkeeper)

        assert bookkeeper.list_actions() == [close, exit_order]
        assert exit_order.reduce_only is False


class TestAnAccountThatDeclaresASymbolTwice:
    async def test_the_surplus_of_an_exit_is_sent_once(self, futures_account):
        strategy_requirements = _declaring(
            [futures_account], [BTC, BTC], with_positions=True
        )
        bookkeeper = _booked_exit(futures_account, PositionSide.LONG, 3.0)

        await split_exits(
            strategy_requirements,
            _holding(futures_account, PositionSide.LONG, 2.0),
            bookkeeper,
        )

        assert _sent(bookkeeper) == [
            (OrderSide.SELL, 2.0, True),
            (OrderSide.SELL, 1.0, False),
        ]


class TestAnExitOnAnAccountThatDeclaredNoPositions:
    @pytest.fixture
    def read_events(self) -> list[str]:
        return []

    @pytest.fixture
    async def first_account(
        self, make_venue: Callable[..., Mock], read_events: list[str]
    ) -> FuturesAccount:
        return await _slow_reading_account(make_venue, "first", read_events)

    @pytest.fixture
    async def second_account(
        self, make_venue: Callable[..., Mock], read_events: list[str]
    ) -> FuturesAccount:
        return await _slow_reading_account(make_venue, "second", read_events)

    async def test_the_position_is_read_and_the_exit_split(
        self, fine_account, fine_venue
    ):
        fine_venue.get_open_positions.return_value = {
            BTC: _position_snapshot(PositionSide.LONG, 0.04)
        }
        bookkeeper = _booked_plain_exit(fine_account, 0.06)

        await _split_undeclared(fine_account, bookkeeper)

        assert _sent(bookkeeper) == [
            (OrderSide.SELL, 0.04, True),
            (OrderSide.SELL, 0.02, False),
        ]
        fine_venue.get_open_positions.assert_awaited_once_with([BTC])

    async def test_one_read_covers_the_symbols_of_all_its_exits(
        self, fine_account, fine_venue
    ):
        fine_venue.get_open_positions.return_value = {}
        strategy_requirements = _declaring([fine_account], [BTC, ETH])
        bookkeeper = _bookkeeper(
            fine_account.long_exit(ETH, 1.0, reduce_only=False).build(),
            _plain_exit(fine_account, 1.0),
        )

        await split_exits(
            strategy_requirements, _account_snapshots(fine_account, None), bookkeeper
        )

        fine_venue.get_open_positions.assert_awaited_once_with([BTC, ETH])

    async def test_a_flat_venue_leaves_the_exit_plain(self, fine_account, fine_venue):
        fine_venue.get_open_positions.return_value = {}
        exit_order = _plain_exit(fine_account, 0.06)
        bookkeeper = _bookkeeper(exit_order)

        await _split_undeclared(fine_account, bookkeeper)

        assert bookkeeper.list_actions() == [exit_order]

    async def test_an_exit_booked_reduce_only_reads_nothing(
        self, fine_account, fine_venue
    ):
        exit_order = fine_account.long_exit(BTC, 0.06).build()
        bookkeeper = _bookkeeper(exit_order)

        await _split_undeclared(fine_account, bookkeeper)

        assert bookkeeper.list_actions() == [exit_order]
        fine_venue.get_open_positions.assert_not_awaited()

    async def test_a_cycle_that_books_no_exit_reads_nothing(
        self, fine_account, fine_venue
    ):
        bookkeeper = _bookkeeper(fine_account.long_entry(BTC, 0.06))

        await _split_undeclared(fine_account, bookkeeper)

        fine_venue.get_open_positions.assert_not_awaited()

    async def test_an_account_that_declared_its_positions_reads_nothing(
        self, futures_account, venue
    ):
        bookkeeper = _booked_exit(futures_account, PositionSide.LONG, 1.0)

        await _split_holding(futures_account, bookkeeper)

        venue.get_open_positions.assert_not_awaited()

    async def test_a_symbol_the_account_declared_no_positions_for_is_read(
        self, futures_account, venue
    ):
        venue.get_open_positions.return_value = {
            ETH: _position_snapshot(PositionSide.LONG, 1.0, ETH)
        }
        bookkeeper = _bookkeeper(
            futures_account.long_exit(ETH, 3.0, reduce_only=False).build()
        )

        await _split_holding(futures_account, bookkeeper)

        assert _sent(bookkeeper) == [
            (OrderSide.SELL, 1.0, True),
            (OrderSide.SELL, 2.0, False),
        ]
        venue.get_open_positions.assert_awaited_once_with([ETH])

    async def test_a_read_that_fails_sends_the_exit_reduce_only_whole_and_says_so(
        self, fine_account, failing_read, caplog
    ):
        bookkeeper = _booked_plain_exit(fine_account, 0.06)

        await _logging_warnings(caplog, _split_undeclared(fine_account, bookkeeper))

        assert _sent(bookkeeper) == [(OrderSide.SELL, 0.06, True)]
        assert "could not be read" in caplog.text
        assert "down" in caplog.text

    async def test_a_read_that_fails_sends_the_member_of_a_batch_reduce_only(
        self, fine_account, failing_read
    ):
        batch = _batch_exit(fine_account, 0.06)
        bookkeeper = _bookkeeper(batch)

        await _split_undeclared(fine_account, bookkeeper)

        (batched,) = bookkeeper.list_actions()
        assert [order.reduce_only for order in batched.orders] == [True]

    async def test_a_read_that_fails_leaves_an_exit_booked_reduce_only_as_booked(
        self, fine_account, failing_read
    ):
        exit_order = fine_account.long_exit(BTC, 0.06).build()
        plain_exit = fine_account.long_exit(ETH, 0.06, reduce_only=False).build()
        bookkeeper = _bookkeeper(exit_order, plain_exit)

        await _split_undeclared(fine_account, bookkeeper)

        assert bookkeeper.list_actions()[0] is exit_order

    async def test_a_transient_failure_is_retried_up_to_the_attempts_given(
        self, fine_account, fine_venue
    ):
        fine_venue.get_open_positions.side_effect = [
            ExchangeTransientError("502"),
            {BTC: _position_snapshot(PositionSide.LONG, 0.04)},
        ]
        bookkeeper = _booked_plain_exit(fine_account, 0.06)

        await _split_undeclared(
            fine_account, bookkeeper, max_attempts=2, base_delay=0.0
        )

        assert _sent(bookkeeper) == [
            (OrderSide.SELL, 0.04, True),
            (OrderSide.SELL, 0.02, False),
        ]
        assert fine_venue.get_open_positions.await_count == 2

    async def test_a_transient_failure_that_outlasts_the_attempts_given_sends_the_exit_reduce_only_whole(
        self, fine_account, fine_venue
    ):
        fine_venue.get_open_positions.side_effect = ExchangeTransientError("502")
        bookkeeper = _booked_plain_exit(fine_account, 0.06)

        await _split_undeclared(
            fine_account, bookkeeper, max_attempts=2, base_delay=0.0
        )

        assert _sent(bookkeeper) == [(OrderSide.SELL, 0.06, True)]
        assert fine_venue.get_open_positions.await_count == 2

    async def test_a_critical_failure_reaches_the_caller(
        self, fine_account, fine_venue
    ):
        fine_venue.get_open_positions.side_effect = ExchangeCriticalError("revoked")
        bookkeeper = _booked_plain_exit(fine_account, 0.06)

        with pytest.raises(ExchangeCriticalError, match="revoked"):
            await _split_undeclared(fine_account, bookkeeper)

    async def test_an_exit_the_venue_rules_deny_asks_for_no_read(
        self, futures_account, venue
    ):
        bookkeeper = _booked_plain_exit(futures_account, 0.25)

        await _split_undeclared(futures_account, bookkeeper)

        venue.get_open_positions.assert_not_awaited()

    async def test_the_accounts_are_read_together(
        self, first_account, second_account, read_events
    ):
        strategy_requirements = _declaring([first_account, second_account], [BTC])
        bookkeeper = _bookkeeper(
            _plain_exit(first_account, 0.06),
            _plain_exit(second_account, 0.06),
        )
        account_snapshots = AccountSnapshots(
            {
                first_account.name: _snapshot_of(first_account, None),
                second_account.name: _snapshot_of(second_account, None),
            }
        )

        await split_exits(strategy_requirements, account_snapshots, bookkeeper)

        assert read_events == [
            "first start",
            "second start",
            "first end",
            "second end",
        ]


class TestWhereTheExitsGoAsBooked:
    async def test_an_account_that_declared_nothing(self, futures_account):
        exit_order = _tracked_exit(futures_account, PositionSide.LONG, 3.0)
        bookkeeper = _bookkeeper(exit_order)

        await split_exits(StrategyRequirements(), AccountSnapshots({}), bookkeeper)

        assert bookkeeper.list_actions() == [exit_order]

    async def test_an_exit_the_venue_rules_deny(self, futures_account):
        denied = _tracked_exit(futures_account, PositionSide.LONG, 0.25)
        bookkeeper = _bookkeeper(denied)

        await _split_holding(futures_account, bookkeeper)

        assert bookkeeper.list_actions() == [denied]
        assert denied.denial is not None

    async def test_a_resting_exit_the_venue_rules_deny(self, futures_account):
        denied = _resting_exit_builder(futures_account, 0.25).build()
        bookkeeper = _bookkeeper(denied)

        await _split_holding(futures_account, bookkeeper)

        assert bookkeeper.list_actions() == [denied]

    async def test_each_account_is_judged_on_its_own_position_and_orders(
        self, futures_account, make_venue
    ):
        second_account = await _read_account(
            make_venue(quantity_step=0.5), [BTC], "second_account"
        )
        strategy_requirements = _declaring(
            [futures_account, second_account], [BTC], with_positions=True
        )
        account_snapshots = _snapshots_of(
            (futures_account, {BTC: _position_snapshot(PositionSide.LONG, 2.0)}),
            (second_account, {}),
        )
        held = _plain_exit(futures_account, 1.0)
        flat = _plain_exit(second_account, 1.0)
        bookkeeper = _bookkeeper(held, flat)

        await split_exits(strategy_requirements, account_snapshots, bookkeeper)

        assert [action.reduce_only for action in bookkeeper.list_actions()] == [
            True,
            False,
        ]


class TestTheRemainderNeverOvertakesItsReducingPart:
    @pytest.fixture
    def placements(self, venue: Mock) -> list[tuple[str, bool]]:
        placed: list[tuple[str, bool]] = []

        async def place_market_order(
            *, reduce_only: bool, **_order: Any
        ) -> PlacedOrder:
            await _recorded(placed, [("start", reduce_only)], ("end", reduce_only))
            return PlacedOrder(order_id=f"order-{len(placed)}")

        async def place_orders(requests: Sequence[Any]) -> list[PlacedOrder]:
            await _recorded(
                placed,
                [("batch start", request.reduce_only) for request in requests],
                ("batch end", True),
            )
            return [PlacedOrder(order_id="batched") for _ in requests]

        venue.place_market_order.side_effect = place_market_order
        venue.place_orders.side_effect = place_orders
        return placed

    async def test_an_exit_booked_to_wait_on_nothing(self, futures_account, placements):
        bookkeeper = _bookkeeper(
            _tracked_exit(futures_account, PositionSide.LONG, 3.0), after=[]
        )
        await _split_holding(futures_account, bookkeeper)

        await _execute(bookkeeper)

        assert placements == [
            ("start", True),
            ("end", True),
            ("start", False),
            ("end", False),
        ]

    async def test_an_exit_in_a_batch(self, futures_account, placements):
        bookkeeper = _bookkeeper(_batch_exit(futures_account), after=[])
        await _split_holding(futures_account, bookkeeper)

        await _execute(bookkeeper)

        assert placements == [
            ("batch start", True),
            ("batch end", True),
            ("start", False),
            ("end", False),
        ]


async def _read_account(
    venue: Mock, symbols: Sequence[Symbol], name: str | None = None
) -> FuturesAccount:
    futures_account = FuturesAccount(venue, name=name)
    await futures_account._read_instruments(symbols)
    return futures_account


def _booked_exit(
    futures_account: FuturesAccount,
    side: PositionSide,
    quantity: float,
    tag: str = "alpha",
) -> BookKeeper:
    return _bookkeeper(_tracked_exit(futures_account, side, quantity, tag))


def _bookkeeper(
    *actions: BaseExchangeAction | FuturesOrderBuilder,
    after: Sequence[BaseExchangeAction] | None = None,
) -> BookKeeper:
    bookkeeper = BookKeeper()
    for action in actions:
        bookkeeper.add(action, after=after)
    return bookkeeper


def _tracked_exit(
    futures_account: FuturesAccount,
    side: PositionSide,
    quantity: float,
    tag: str = "alpha",
) -> BatchableOrderAction:
    return (
        _tracked_builder(futures_account, side, quantity)
        .tag(tag)
        .reason("exit")
        .build()
    )


def _tracked_builder(
    futures_account: FuturesAccount, side: PositionSide, quantity: float
) -> FuturesOrderBuilder:
    return futures_account.close_tracked_position(BTC, TrackedPosition(side, quantity))


async def _split_holding(
    futures_account: FuturesAccount,
    bookkeeper: BookKeeper,
    side: PositionSide = PositionSide.LONG,
    quantity: float = 2.0,
) -> None:
    await _split(futures_account, bookkeeper, _holding(futures_account, side, quantity))


async def _split(
    futures_account: FuturesAccount,
    bookkeeper: BookKeeper,
    account_snapshots: AccountSnapshots,
    with_positions: bool = True,
    **retries: Any,
) -> None:
    await split_exits(
        _declared(futures_account, with_positions),
        account_snapshots,
        bookkeeper,
        **retries,
    )


def _declared(
    futures_account: FuturesAccount, with_positions: bool = True
) -> StrategyRequirements:
    return _declaring(
        [futures_account], [BTC], with_positions=with_positions, open_orders=True
    )


def _declaring(
    accounts: Iterable[FuturesAccount],
    symbols: Sequence[Symbol],
    with_positions: bool = False,
    open_orders: bool = False,
) -> StrategyRequirements:
    strategy_requirements = StrategyRequirements()
    for futures_account in accounts:
        strategy_requirements.account.add(
            futures_account,
            symbols=symbols,
            positions=with_positions,
            open_orders=open_orders,
        )
    return strategy_requirements


def _holding(
    futures_account: FuturesAccount, side: PositionSide, quantity: float
) -> AccountSnapshots:
    return _account_snapshots(
        futures_account, {BTC: _position_snapshot(side, quantity)}
    )


def _account_snapshots(
    futures_account: FuturesAccount,
    held_positions: dict[Symbol, PositionSnapshot] | None,
) -> AccountSnapshots:
    return _snapshots_of((futures_account, held_positions))


def _snapshots_of(
    *held: tuple[FuturesAccount, dict[Symbol, PositionSnapshot] | None],
) -> AccountSnapshots:
    return AccountSnapshots(
        {
            futures_account.name: _snapshot_of(futures_account, positions)
            for futures_account, positions in held
        }
    )


def _snapshot_of(
    futures_account: FuturesAccount,
    held_positions: dict[Symbol, PositionSnapshot] | None,
) -> AccountSnapshot:
    return AccountSnapshot(
        account_name=futures_account.name,
        positions=held_positions,
        open_orders=[],
        executions=[],
    )


def _position_snapshot(
    side: PositionSide, quantity: float, symbol: Symbol = BTC
) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        side=side,
        quantity=quantity,
        average_entry_price=100.0,
        entry_time=datetime(2026, 1, 1, tzinfo=UTC),
        leverage=1.0,
        liquidation_price=1.0,
        stop_loss_price=None,
        take_profit_price=None,
    )


def _sent(bookkeeper: BookKeeper) -> list[tuple[str, float, bool]]:
    return _sides_and_quantities(bookkeeper.list_actions())


def _sides_and_quantities(
    actions: Iterable[Any],
) -> list[tuple[str, float, bool]]:
    return [(action.side, action.quantity, action.reduce_only) for action in actions]


def _resting_exit(
    futures_account: FuturesAccount, quantity: float, tag: str
) -> BatchableOrderAction:
    return _resting_exit_builder(futures_account, quantity).tag(tag).build()


def _resting_exit_builder(
    futures_account: FuturesAccount, quantity: float
) -> FuturesOrderBuilder:
    return futures_account.long_exit(BTC, quantity, reduce_only=False).limit(110.0)


async def _ignore_fill(_placed: object, _fill: object) -> None:
    return None


def _sent_without_closes(bookkeeper: BookKeeper) -> list[tuple[str, float, bool]]:
    return _sides_and_quantities(
        action
        for action in bookkeeper.list_actions()
        if not isinstance(action, ClosePositionAction)
    )


def _batch_exit(
    futures_account: FuturesAccount, quantity: float = 3.0
) -> BaseExchangeAction:
    return futures_account.place_orders(
        [futures_account.long_exit(BTC, quantity, reduce_only=False).tag("a")]
    )


async def _split_flat(futures_account: FuturesAccount, bookkeeper: BookKeeper) -> None:
    await _split(futures_account, bookkeeper, _account_snapshots(futures_account, {}))


def _booked_reduce_only_exit(
    futures_account: FuturesAccount, quantity: float, tag: str
) -> BookKeeper:
    return _bookkeeper(_reduce_only_exit(futures_account, quantity, tag))


def _reduce_only_exit(
    futures_account: FuturesAccount, quantity: float, tag: str
) -> BatchableOrderAction:
    return futures_account.long_exit(BTC, quantity).tag(tag).build()


async def _logging_warnings(
    caplog: pytest.LogCaptureFixture, split: Awaitable[None]
) -> None:
    with caplog.at_level(logging.WARNING):
        await split


def _target_snapshot(futures_account: FuturesAccount) -> AccountSnapshot:
    return AccountSnapshot(
        account_name=futures_account.name,
        balances={"USDT": Balance(locked=0.0, total=10_000.0)},
        positions={BTC: _position_snapshot(PositionSide.LONG, 2.0)},
        conversion_rates={BTC: 1.0},
    )


def _booking_replacement(
    futures_account: FuturesAccount, replacement: FuturesOrderBuilder
) -> BookKeeper:
    return _bookkeeper(futures_account.modify_order("resting-1", replacement))


def _track_alpha(
    strategy_requirements: StrategyRequirements,
    futures_account: FuturesAccount,
    account_snapshots: AccountSnapshots,
) -> None:
    strategy_requirements.tracker.add(
        futures_account,
        symbols={ALPHA: BTC},
        tags={ALPHA: profile_tag("1d", "alpha")},
    )
    strategy_requirements.tracker.refresh_all(
        account_snapshots, datetime.min, datetime.max
    )


async def _slow_reading_account(
    make_venue: Callable[..., Mock], name: str, events: list[str]
) -> FuturesAccount:
    venue = make_venue()
    venue.get_open_positions.side_effect = _slow_read(events, name)
    return await _read_account(venue, [BTC], name)


def _slow_read(
    events: list[str], name: str
) -> Callable[[list[Symbol]], Awaitable[dict[Symbol, PositionSnapshot]]]:
    async def get_open_positions(
        _symbols: list[Symbol],
    ) -> dict[Symbol, PositionSnapshot]:
        await _recorded(events, [f"{name} start"], f"{name} end")
        return {}

    return get_open_positions


async def _recorded(events: list[Any], starts: Sequence[Any], end: Any) -> None:
    events.extend(starts)
    await asyncio.sleep(0.01)
    events.append(end)


def _booked_plain_exit(futures_account: FuturesAccount, quantity: float) -> BookKeeper:
    return _bookkeeper(_plain_exit(futures_account, quantity))


def _plain_exit(
    futures_account: FuturesAccount, quantity: float
) -> BatchableOrderAction:
    return futures_account.long_exit(BTC, quantity, reduce_only=False).build()


async def _split_undeclared(
    futures_account: FuturesAccount, bookkeeper: BookKeeper, **retries: Any
) -> None:
    await _split(
        futures_account,
        bookkeeper,
        _account_snapshots(futures_account, None),
        with_positions=False,
        **retries,
    )


async def _execute(bookkeeper: BookKeeper) -> None:
    await execute_trading_actions(
        bookkeeper.list_actions(), declared_waits=bookkeeper.declared_waits()
    )
