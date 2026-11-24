import logging
from collections.abc import Callable
from datetime import datetime, timezone

import pytest

from robottraderslab_core import Symbol
from robottraderslab_core._internal.account_snapshot import AccountSnapshot
from robottraderslab_core._internal.actions.members.tags import profile_tag
from robottraderslab_core._internal.exchange_position_tracker import (
    ExchangePositionTracker,
)
from robottraderslab_core._internal.position_tracker import TrackedPosition, TrackingId
from robottraderslab_core.exceptions import BotCriticalError
from robottraderslab_core.exchanges import (
    Execution,
    FillEffect,
    OrderSide,
    OrderType,
    PositionSide,
    PositionSnapshot,
    client_order_id_carrying,
)

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")
ALPHA = TrackingId("BTC/USDT:USDT@1d#alpha")
BETA = TrackingId("BTC/USDT:USDT@1d#beta")

T0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
READ_SINCE = datetime(2025, 10, 3, tzinfo=timezone.utc)
CANDLE_OPEN = T0
CANDLE_CLOSE = datetime(2026, 1, 2, tzinfo=timezone.utc)
A_LATER_CANDLE_OPEN = CANDLE_CLOSE
A_LATER_CANDLE_CLOSE = datetime(2026, 1, 3, tzinfo=timezone.utc)
THE_NEXT_CANDLE_OPEN = A_LATER_CANDLE_CLOSE
THE_NEXT_CANDLE_CLOSE = datetime(2026, 1, 4, tzinfo=timezone.utc)


@pytest.fixture
def create_execution() -> Callable[..., Execution]:
    def _create_execution(
        *,
        execution_id: str,
        side: OrderSide,
        quantity: float,
        timestamp: datetime,
        effect: FillEffect,
        tag: str | None = None,
        symbol: Symbol = BTC,
        kind: OrderType | None = None,
    ) -> Execution:
        return Execution(
            execution_id=execution_id,
            order_id=execution_id,
            symbol=symbol,
            side=side,
            price=100.0,
            quantity=quantity,
            timestamp=timestamp,
            effect=effect,
            kind=kind,
            client_order_id=client_order_id_carrying(tag) if tag is not None else None,
        )

    return _create_execution


@pytest.fixture
def create_snapshot() -> Callable[[list[Execution], float | None], AccountSnapshot]:
    def _create_snapshot(
        executions: list[Execution], venue_quantity: float | None
    ) -> AccountSnapshot:
        positions = (
            {}
            if venue_quantity is None
            else {
                BTC: PositionSnapshot(
                    symbol=BTC,
                    side=(
                        PositionSide.LONG if venue_quantity >= 0 else PositionSide.SHORT
                    ),
                    quantity=abs(venue_quantity),
                    average_entry_price=100.0,
                    entry_time=T0,
                    leverage=1.0,
                    liquidation_price=0.0,
                    stop_loss_price=None,
                    take_profit_price=None,
                )
            }
        )
        return AccountSnapshot(
            account_name="test",
            positions=positions,
            executions=executions,
            executions_declared=True,
        )

    return _create_snapshot


@pytest.fixture
def tracker() -> ExchangePositionTracker:
    return _tracker()


class TestCreateValidation:
    def test_an_id_left_without_a_tag_is_refused(self):
        with pytest.raises(BotCriticalError, match="have no tag"):
            ExchangePositionTracker.create(
                {BTC: (ALPHA, BETA)},
                {ALPHA: profile_tag("1h", "a")},
            )

    def test_two_ids_sharing_a_tag_on_one_symbol_are_refused(self):
        with pytest.raises(BotCriticalError, match="share the tag"):
            ExchangePositionTracker.create(
                {BTC: (ALPHA, BETA)},
                {ALPHA: profile_tag("1h", "same"), BETA: profile_tag("1h", "same")},
            )

    def test_the_same_tag_on_two_symbols_is_accepted(self):
        eth_id = TrackingId("ETH/USDT:USDT@1d#alpha")

        tracker = ExchangePositionTracker.create(
            {BTC: (ALPHA,), ETH: (eth_id,)},
            {ALPHA: profile_tag("1h", "same"), eth_id: profile_tag("1h", "same")},
        )

        assert tracker.get(ALPHA) is None


class TestFoldingTaggedFills:
    def test_a_tagged_buy_opens_exposure(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=3.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )

        tracker.refresh(
            create_snapshot([opened], venue_quantity=3.0), CANDLE_OPEN, CANDLE_CLOSE
        )

        position = tracker.get(ALPHA)
        assert position.side == PositionSide.LONG
        assert position.quantity == 3.0

    def test_a_tagged_sell_reduces_exposure(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=5.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        reduced = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=2.0,
            timestamp=T1,
            effect="reduce",
            tag="1h-a",
        )

        tracker.refresh(
            create_snapshot([opened, reduced], venue_quantity=3.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA).quantity == 3.0

    def test_a_short_side_tag_accumulates_negative_exposure(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.SELL,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )

        tracker.refresh(
            create_snapshot([opened], venue_quantity=-4.0), CANDLE_OPEN, CANDLE_CLOSE
        )

        position = tracker.get(ALPHA)
        assert position.side == PositionSide.SHORT
        assert position.quantity == 4.0

    def test_an_untagged_fill_credits_no_tracking_id(
        self, tracker, create_execution, create_snapshot
    ):
        untagged = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=5.0,
            timestamp=T0,
            effect="increase",
        )

        tracker.refresh(
            create_snapshot([untagged], venue_quantity=5.0), CANDLE_OPEN, CANDLE_CLOSE
        )

        assert tracker.get(ALPHA) is None
        assert tracker.get(BETA) is None

    def test_each_tag_is_credited_to_its_own_id(
        self, tracker, create_execution, create_snapshot
    ):
        alpha_fill = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        beta_fill = create_execution(
            execution_id="e2",
            side=OrderSide.BUY,
            quantity=6.0,
            timestamp=T0,
            effect="increase",
            tag="1h-b",
        )

        tracker.refresh(
            create_snapshot([alpha_fill, beta_fill], venue_quantity=10.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA).quantity == 4.0
        assert tracker.get(BETA).quantity == 6.0

    def test_a_tag_shared_across_symbols_credits_the_fills_own_symbol(
        self, create_execution, create_snapshot
    ):
        eth_id = TrackingId("ETH/USDT:USDT@1d#alpha")
        tracker = ExchangePositionTracker.create(
            {BTC: (ALPHA,), ETH: (eth_id,)},
            {ALPHA: profile_tag("1h", "same"), eth_id: profile_tag("1h", "same")},
        )
        eth_fill = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=2.0,
            timestamp=T0,
            effect="open",
            tag="1h-same",
            symbol=ETH,
        )

        tracker.refresh(
            create_snapshot([eth_fill], venue_quantity=None), CANDLE_OPEN, CANDLE_CLOSE
        )

        assert tracker.get(ALPHA) is None
        assert tracker.get(eth_id).quantity == 2.0


class TestClosingFills:
    def test_a_tagged_close_takes_only_the_id_that_placed_it_to_zero(
        self, tracker, create_execution, create_snapshot
    ):
        alpha_fill = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        beta_fill = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=4.0,
            timestamp=T1,
            effect="close",
            tag="1h-b",
        )

        tracker.refresh(
            create_snapshot([alpha_fill, beta_fill], venue_quantity=None),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 4.0)
        assert tracker.get(BETA) == TrackedPosition(PositionSide.SHORT, 4.0)

    def test_an_untagged_close_still_clears_every_id(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        tracker.refresh(
            create_snapshot([opened], venue_quantity=4.0), CANDLE_OPEN, CANDLE_CLOSE
        )

        stop_loss = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=4.0,
            timestamp=T1,
            effect="close",
        )
        tracker.refresh(
            create_snapshot([opened, stop_loss], venue_quantity=None),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) is None


class TestNettedDeclaration:
    @pytest.fixture
    def netted_tracker(self) -> ExchangePositionTracker:
        return _tracker(netted=True)

    def test_tagged_fills_netting_the_venue_to_flat_leave_no_id_holding(
        self, netted_tracker, create_execution, create_snapshot
    ):
        bought = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        sold = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=4.0,
            timestamp=T1,
            effect="close",
            tag="1h-b",
        )

        netted_tracker.refresh(
            create_snapshot([bought, sold], venue_quantity=None),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert netted_tracker.get(ALPHA) is None
        assert netted_tracker.get(BETA) is None

    def test_a_tagged_fill_crossing_flat_holds_the_remainder_on_its_own_id(
        self, netted_tracker, create_execution, create_snapshot
    ):
        bought = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        sold = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=6.0,
            timestamp=T1,
            effect="reverse",
            tag="1h-b",
        )

        netted_tracker.refresh(
            create_snapshot([bought, sold], venue_quantity=-2.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert netted_tracker.get(ALPHA) is None
        assert netted_tracker.get(BETA) == TrackedPosition(PositionSide.SHORT, 2.0)

    def test_tagged_fills_never_netting_the_venue_to_flat_each_keep_their_side(
        self, netted_tracker, create_execution, create_snapshot
    ):
        bought = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        sold = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=3.0,
            timestamp=T1,
            effect="reduce",
            tag="1h-b",
        )

        netted_tracker.refresh(
            create_snapshot([bought, sold], venue_quantity=1.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert netted_tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 4.0)
        assert netted_tracker.get(BETA) == TrackedPosition(PositionSide.SHORT, 3.0)

    def test_a_tagged_fill_taking_the_venue_through_flat_reports_the_outside_share_gone(
        self, netted_tracker, create_execution, create_snapshot, caplog
    ):
        by_hand = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=2.0,
            timestamp=T0,
            effect="open",
        )
        sold = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=2.0,
            timestamp=T1,
            effect="close",
            tag="1h-a",
        )

        with caplog.at_level(logging.INFO):
            netted_tracker.refresh(
                create_snapshot([by_hand, sold], venue_quantity=None),
                CANDLE_OPEN,
                CANDLE_CLOSE,
            )

        assert [record.levelname for record in caplog.records] == ["WARNING"]
        assert "is gone" in caplog.text


class TestReEntryAfterAClose:
    def test_a_later_tagged_open_is_credited_after_a_close(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        closed = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=4.0,
            timestamp=T1,
            effect="close",
            tag="1h-a",
        )
        tracker.refresh(
            create_snapshot([opened], venue_quantity=4.0), CANDLE_OPEN, CANDLE_CLOSE
        )
        tracker.refresh(
            create_snapshot([opened, closed], venue_quantity=None),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        reopened = create_execution(
            execution_id="e3",
            side=OrderSide.SELL,
            quantity=2.0,
            timestamp=T2,
            effect="open",
            tag="1h-a",
        )
        tracker.refresh(
            create_snapshot([opened, closed, reopened], venue_quantity=-2.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        position = tracker.get(ALPHA)
        assert position.side == PositionSide.SHORT
        assert position.quantity == 2.0


class TestIncrementalRefresh:
    def test_the_same_snapshot_refreshed_twice_leaves_exposure_unchanged(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        snapshot = create_snapshot([opened], venue_quantity=4.0)
        tracker.refresh(snapshot, CANDLE_OPEN, CANDLE_CLOSE)

        tracker.refresh(snapshot, CANDLE_OPEN, CANDLE_CLOSE)

        assert tracker.get(ALPHA).quantity == 4.0

    def test_a_grown_stream_includes_its_new_tail(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        tracker.refresh(
            create_snapshot([opened], venue_quantity=4.0), CANDLE_OPEN, CANDLE_CLOSE
        )

        increased = create_execution(
            execution_id="e2",
            side=OrderSide.BUY,
            quantity=3.0,
            timestamp=T1,
            effect="increase",
            tag="1h-a",
        )
        tracker.refresh(
            create_snapshot([opened, increased], venue_quantity=7.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA).quantity == 7.0

    def test_an_execution_aged_out_of_the_fetched_window_is_not_forgotten(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        tracker.refresh(
            create_snapshot([opened], venue_quantity=4.0), CANDLE_OPEN, CANDLE_CLOSE
        )

        increased = create_execution(
            execution_id="e2",
            side=OrderSide.BUY,
            quantity=3.0,
            timestamp=T1,
            effect="increase",
            tag="1h-a",
        )
        tracker.refresh(
            create_snapshot([increased], venue_quantity=7.0), CANDLE_OPEN, CANDLE_CLOSE
        )

        assert tracker.get(ALPHA).quantity == 7.0

    def test_out_of_order_executions_still_fold_oldest_first(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        closed = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=4.0,
            timestamp=T1,
            effect="close",
            tag="1h-a",
        )

        tracker.refresh(
            create_snapshot([closed, opened], venue_quantity=None),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) is None

    def test_two_fills_sharing_a_timestamp_are_each_folded_once(
        self, tracker, create_execution, create_snapshot
    ):
        first = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=1.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        tracker.refresh(
            create_snapshot([first], venue_quantity=1.0), CANDLE_OPEN, CANDLE_CLOSE
        )

        second = create_execution(
            execution_id="e2",
            side=OrderSide.BUY,
            quantity=2.0,
            timestamp=T0,
            effect="increase",
            tag="1h-a",
        )
        tracker.refresh(
            create_snapshot([first, second], venue_quantity=3.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        tracker.refresh(
            create_snapshot([first, second], venue_quantity=3.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA).quantity == 3.0

    def test_a_fill_landing_late_at_the_latest_timestamp_with_a_lower_id_is_folded_once(
        self, tracker, create_execution, create_snapshot
    ):
        folded = create_execution(
            execution_id="e2",
            side=OrderSide.BUY,
            quantity=1.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        tracker.refresh(
            create_snapshot([folded], venue_quantity=1.0), CANDLE_OPEN, CANDLE_CLOSE
        )
        landed_late = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=2.0,
            timestamp=T0,
            effect="increase",
            tag="1h-a",
        )

        tracker.refresh(
            create_snapshot([landed_late, folded], venue_quantity=3.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )
        tracker.refresh(
            create_snapshot([landed_late, folded], venue_quantity=3.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA).quantity == 3.0


class TestQuietCandle:
    def test_a_candle_with_no_new_execution_leaves_what_the_candle_before_left(
        self, tracker, create_execution, create_snapshot, caplog
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        by_hand = create_execution(
            execution_id="e2",
            side=OrderSide.BUY,
            quantity=2.0,
            timestamp=T1,
            effect="increase",
        )
        tracker.refresh(
            create_snapshot([opened, by_hand], venue_quantity=6.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        with caplog.at_level(logging.INFO):
            asking = tracker.refresh(
                create_snapshot([opened, by_hand], venue_quantity=6.0),
                A_LATER_CANDLE_OPEN,
                A_LATER_CANDLE_CLOSE,
            )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 4.0)
        assert tracker.get(BETA) is None
        assert not asking
        assert tracker.unanchored_symbols() == frozenset()
        assert "+2 of the venue's position is held outside" in caplog.text

    def test_a_venue_position_moving_with_no_new_execution_is_still_seen(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        tracker.refresh(
            create_snapshot([opened], venue_quantity=4.0), CANDLE_OPEN, CANDLE_CLOSE
        )

        moved = tracker.refresh(
            create_snapshot([opened], venue_quantity=7.0), CANDLE_OPEN, CANDLE_CLOSE
        )
        assert moved
        assert tracker.unanchored_symbols() == frozenset({BTC})

        settled = tracker.refresh(
            create_snapshot([opened], venue_quantity=4.0), CANDLE_OPEN, CANDLE_CLOSE
        )
        assert not settled
        assert tracker.unanchored_symbols() == frozenset()

    def test_a_position_read_empty_after_a_held_one_is_unanchored_as_on_a_flat_read(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        tracker.refresh(
            create_snapshot([opened], venue_quantity=4.0), CANDLE_OPEN, CANDLE_CLOSE
        )

        asking = tracker.refresh(
            create_snapshot([opened], venue_quantity=None),
            A_LATER_CANDLE_OPEN,
            A_LATER_CANDLE_CLOSE,
        )

        assert asking
        assert tracker.unanchored_symbols() == frozenset({BTC})

    def test_a_candle_with_no_execution_at_all_leaves_nothing_held(
        self, tracker, create_snapshot
    ):
        tracker.refresh(
            create_snapshot([], venue_quantity=None), CANDLE_OPEN, CANDLE_CLOSE
        )

        asking = tracker.refresh(
            create_snapshot([], venue_quantity=None),
            A_LATER_CANDLE_OPEN,
            A_LATER_CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) is None
        assert not asking
        assert tracker.unanchored_symbols() == frozenset()

    def test_a_fill_on_a_symbol_the_tracker_does_not_track_is_ignored(
        self, tracker, create_execution, create_snapshot
    ):
        tracked = create_execution(
            execution_id="b1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        untracked = create_execution(
            execution_id="x1",
            side=OrderSide.BUY,
            quantity=9.0,
            timestamp=T1,
            effect="open",
            tag="1h-a",
            symbol=ETH,
        )

        asking = tracker.refresh(
            create_snapshot([tracked, untracked], venue_quantity=4.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 4.0)
        assert not asking
        assert tracker.unanchored_symbols() == frozenset()

    def test_executions_landing_mid_read_are_folded_whatever_the_ends_of_the_read(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="b1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        eth_fills = [
            create_execution(
                execution_id=f"x{number}",
                side=OrderSide.BUY,
                quantity=1.0,
                timestamp=T0,
                effect="open",
                symbol=ETH,
            )
            for number in (1, 2, 3)
        ]
        tracker.refresh(
            create_snapshot([opened, *eth_fills], venue_quantity=4.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )
        alpha_buy = create_execution(
            execution_id="b2",
            side=OrderSide.BUY,
            quantity=1.0,
            timestamp=T1,
            effect="increase",
            tag="1h-a",
        )
        beta_sell = create_execution(
            execution_id="b3",
            side=OrderSide.SELL,
            quantity=1.0,
            timestamp=T2,
            effect="decrease",
            tag="1h-b",
        )

        tracker.refresh(
            create_snapshot(
                [opened, alpha_buy, beta_sell, eth_fills[-1]], venue_quantity=4.0
            ),
            A_LATER_CANDLE_OPEN,
            A_LATER_CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 5.0)
        assert tracker.get(BETA) == TrackedPosition(PositionSide.SHORT, 1.0)

    def test_a_candle_after_a_widened_read_keeps_what_the_wider_fold_settled(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e2",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T1,
            effect="open",
            tag="1h-a",
        )
        by_hand = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=6.0,
            timestamp=T0,
            effect="open",
        )
        snapshot = create_snapshot([opened], venue_quantity=10.0)
        tracker.refresh(snapshot, CANDLE_OPEN, CANDLE_CLOSE)
        tracker.widen(
            snapshot,
            [by_hand, opened],
            tracker.unanchored_symbols(),
            READ_SINCE,
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        asking = tracker.refresh(
            create_snapshot([opened], venue_quantity=10.0),
            A_LATER_CANDLE_OPEN,
            A_LATER_CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 4.0)
        assert not asking
        assert tracker.unanchored_symbols() == frozenset()


class TestAFillLandingAfterThePositionRead:
    def test_the_symbol_is_unanchored_until_the_position_read_catches_up(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        by_hand = create_execution(
            execution_id="e2",
            side=OrderSide.BUY,
            quantity=2.0,
            timestamp=T1,
            effect="increase",
        )
        tracker.refresh(
            create_snapshot([opened, by_hand], venue_quantity=6.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )
        landed = create_execution(
            execution_id="e3",
            side=OrderSide.BUY,
            quantity=1.0,
            timestamp=T2,
            effect="increase",
            tag="1h-a",
        )

        asking = tracker.refresh(
            create_snapshot([opened, by_hand, landed], venue_quantity=6.0),
            A_LATER_CANDLE_OPEN,
            A_LATER_CANDLE_CLOSE,
        )

        assert asking
        assert tracker.unanchored_symbols() == frozenset({BTC})

    def test_the_share_held_outside_is_what_it_was_once_the_position_read_catches_up(
        self, tracker, create_execution, create_snapshot, caplog
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        by_hand = create_execution(
            execution_id="e2",
            side=OrderSide.BUY,
            quantity=2.0,
            timestamp=T1,
            effect="increase",
        )
        landed = create_execution(
            execution_id="e3",
            side=OrderSide.BUY,
            quantity=1.0,
            timestamp=T2,
            effect="increase",
            tag="1h-a",
        )
        tracker.refresh(
            create_snapshot([opened, by_hand], venue_quantity=6.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )
        tracker.refresh(
            create_snapshot([opened, by_hand, landed], venue_quantity=6.0),
            A_LATER_CANDLE_OPEN,
            A_LATER_CANDLE_CLOSE,
        )

        with caplog.at_level(logging.INFO):
            asking = tracker.refresh(
                create_snapshot([opened, by_hand, landed], venue_quantity=7.0),
                THE_NEXT_CANDLE_OPEN,
                THE_NEXT_CANDLE_CLOSE,
            )

        assert not asking
        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 5.0)
        assert "+2 of the venue's position is held outside" in caplog.text


class TestExactToTheStep:
    def test_a_partial_exit_leaves_exactly_the_step_its_fills_add_up_to(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=0.00001037,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        reduced = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=0.00001036,
            timestamp=T1,
            effect="reduce",
            tag="1h-a",
        )

        tracker.refresh(
            create_snapshot([opened, reduced], venue_quantity=1e-8),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 1e-8)

    def test_a_fill_under_half_a_step_reads_as_flat(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4e-9,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )

        tracker.refresh(
            create_snapshot([opened], venue_quantity=4e-9), CANDLE_OPEN, CANDLE_CLOSE
        )

        assert tracker.get(ALPHA) is None


class TestUnownedQuantity:
    def test_a_position_the_read_does_not_explain_waits_for_the_wider_read(
        self, tracker, create_execution, create_snapshot, caplog
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )

        with caplog.at_level(logging.INFO):
            tracker.refresh(
                create_snapshot([opened], venue_quantity=10.0),
                CANDLE_OPEN,
                CANDLE_CLOSE,
            )

        assert caplog.text == ""
        assert tracker.unanchored_symbols() == frozenset({BTC})

    def test_a_wider_read_reaching_a_flat_moment_attributes_the_whole_position(
        self, tracker, create_execution, create_snapshot, caplog
    ):
        opened = create_execution(
            execution_id="e2",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T1,
            effect="open",
            tag="1h-a",
        )
        by_hand = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=6.0,
            timestamp=T0,
            effect="open",
        )
        snapshot = create_snapshot([opened], venue_quantity=10.0)
        tracker.refresh(snapshot, CANDLE_OPEN, CANDLE_CLOSE)

        with caplog.at_level(logging.WARNING):
            tracker.widen(
                snapshot,
                [by_hand, opened],
                tracker.unanchored_symbols(),
                READ_SINCE,
                CANDLE_OPEN,
                CANDLE_CLOSE,
            )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 4.0)
        assert tracker.unanchored_symbols() == frozenset()
        assert "+6" in caplog.text
        assert "2026-01-01 00:00 UTC" in caplog.text

    def test_quantity_older_than_every_fill_the_venue_keeps_names_the_oldest(
        self, tracker, create_execution, create_snapshot, caplog
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T1,
            effect="open",
            tag="1h-a",
        )
        snapshot = create_snapshot([opened], venue_quantity=10.0)
        tracker.refresh(snapshot, CANDLE_OPEN, CANDLE_CLOSE)

        with caplog.at_level(logging.WARNING):
            tracker.widen(
                snapshot,
                [opened],
                tracker.unanchored_symbols(),
                READ_SINCE,
                CANDLE_OPEN,
                CANDLE_CLOSE,
            )

        assert "+6" in caplog.text
        assert "2025-10-03 00:00 UTC" in caplog.text
        assert "left unattributed" in caplog.text

    def test_an_untagged_fill_of_the_closed_candle_names_what_it_opened(
        self, tracker, create_execution, create_snapshot, caplog
    ):
        by_hand = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=6.0,
            timestamp=T0,
            effect="open",
        )

        with caplog.at_level(logging.INFO):
            tracker.refresh(
                create_snapshot([by_hand], venue_quantity=6.0),
                CANDLE_OPEN,
                CANDLE_CLOSE,
            )

        assert [record.levelname for record in caplog.records] == ["WARNING"]
        assert "+6" in caplog.text
        assert "2026-01-01 00:00 UTC" in caplog.text

    def test_a_quantity_standing_from_an_earlier_candle_is_stated_and_not_warned(
        self, create_execution, create_snapshot, caplog
    ):
        by_hand = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=6.0,
            timestamp=T0,
            effect="open",
        )
        snapshot = create_snapshot([by_hand], venue_quantity=6.0)

        with caplog.at_level(logging.INFO):
            _tracker().refresh(snapshot, A_LATER_CANDLE_OPEN, A_LATER_CANDLE_CLOSE)

        assert [record.levelname for record in caplog.records] == ["INFO"]
        assert "+6" in caplog.text

    def test_a_fill_landing_after_the_close_is_the_next_candles_to_report(
        self, tracker, create_execution, create_snapshot, caplog
    ):
        after_the_close = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=6.0,
            timestamp=CANDLE_CLOSE,
            effect="open",
        )
        snapshot = create_snapshot([after_the_close], venue_quantity=6.0)

        with caplog.at_level(logging.INFO):
            tracker.refresh(snapshot, CANDLE_OPEN, CANDLE_CLOSE)
            _tracker().refresh(snapshot, A_LATER_CANDLE_OPEN, A_LATER_CANDLE_CLOSE)

        assert [record.levelname for record in caplog.records] == ["INFO", "WARNING"]

    def test_a_hand_closing_the_symbol_in_the_closed_candle_reports_it_gone(
        self, tracker, create_execution, create_snapshot, caplog
    ):
        by_hand = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=6.0,
            timestamp=T0,
            effect="open",
        )
        sold_by_hand = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=6.0,
            timestamp=T1,
            effect="close",
        )

        with caplog.at_level(logging.INFO):
            tracker.refresh(
                create_snapshot([by_hand, sold_by_hand], venue_quantity=None),
                CANDLE_OPEN,
                CANDLE_CLOSE,
            )

        assert [record.levelname for record in caplog.records] == ["WARNING"]
        assert "is gone" in caplog.text

    def test_a_liquidation_of_what_the_ids_held_reports_nothing(
        self, tracker, create_execution, create_snapshot, caplog
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=6.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        liquidated = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=6.0,
            timestamp=T1,
            effect="close",
        )

        with caplog.at_level(logging.INFO):
            tracker.refresh(
                create_snapshot([opened, liquidated], venue_quantity=None),
                CANDLE_OPEN,
                CANDLE_CLOSE,
            )

        assert caplog.text == ""
        assert tracker.get(ALPHA) is None

    def test_a_symbol_flat_since_an_earlier_candle_says_nothing(
        self, tracker, create_execution, create_snapshot, caplog
    ):
        by_hand = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=6.0,
            timestamp=T0,
            effect="open",
        )
        sold_by_hand = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=6.0,
            timestamp=T1,
            effect="close",
        )

        with caplog.at_level(logging.INFO):
            tracker.refresh(
                create_snapshot([by_hand, sold_by_hand], venue_quantity=None),
                A_LATER_CANDLE_OPEN,
                A_LATER_CANDLE_CLOSE,
            )

        assert caplog.text == ""

    def test_an_untagged_reversal_leaves_the_far_side_outside_every_id(
        self, tracker, create_execution, create_snapshot
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        reversed_by_hand = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=6.0,
            timestamp=T1,
            effect="close",
        )

        tracker.refresh(
            create_snapshot([opened, reversed_by_hand], venue_quantity=-2.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) is None
        assert tracker.unanchored_symbols() == frozenset()

    def test_a_position_within_the_read_is_never_reported(
        self, tracker, create_execution, create_snapshot, caplog
    ):
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=10.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )

        with caplog.at_level(logging.INFO):
            tracker.refresh(
                create_snapshot([opened], venue_quantity=10.0),
                CANDLE_OPEN,
                CANDLE_CLOSE,
            )

        assert caplog.text == ""


class TestFiredProtectionWithoutLabel:
    @pytest.fixture
    def opened_by_alpha(
        self,
        tracker: ExchangePositionTracker,
        create_execution: Callable[..., Execution],
        create_snapshot: Callable[[list[Execution], float | None], AccountSnapshot],
    ) -> Execution:
        opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=10.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        tracker.refresh(
            create_snapshot([opened], venue_quantity=10.0), CANDLE_OPEN, CANDLE_CLOSE
        )
        return opened

    def test_a_stop_loss_closing_part_of_the_only_long_is_credited_to_it(
        self, tracker, create_execution, create_snapshot, opened_by_alpha, caplog
    ):
        stop_loss = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=4.0,
            timestamp=T1,
            effect="close",
            kind=OrderType.STOP_LOSS,
        )

        with caplog.at_level(logging.INFO):
            tracker.refresh(
                create_snapshot([opened_by_alpha, stop_loss], venue_quantity=6.0),
                CANDLE_OPEN,
                CANDLE_CLOSE,
            )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 6.0)
        assert caplog.records == []

    def test_a_take_profit_carrying_the_engines_tag_is_credited_the_same(
        self, tracker, create_execution, create_snapshot, opened_by_alpha
    ):
        take_profit = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=4.0,
            timestamp=T1,
            effect="close",
            kind=OrderType.TAKE_PROFIT,
            tag="tp",
        )

        tracker.refresh(
            create_snapshot([opened_by_alpha, take_profit], venue_quantity=6.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 6.0)

    def test_a_buy_stop_closes_the_short_and_leaves_the_long(
        self, tracker, create_execution, create_snapshot
    ):
        long_opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=10.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        short_opened = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=3.0,
            timestamp=T1,
            effect="open",
            tag="1h-b",
        )
        stop_loss = create_execution(
            execution_id="e3",
            side=OrderSide.BUY,
            quantity=3.0,
            timestamp=T2,
            effect="close",
            kind=OrderType.STOP_LOSS,
        )

        tracker.refresh(
            create_snapshot(
                [long_opened, short_opened, stop_loss], venue_quantity=10.0
            ),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 10.0)
        assert tracker.get(BETA) is None

    def test_several_ids_on_the_closed_side_resolve_to_the_one_the_quantity_equals(
        self, tracker, create_execution, create_snapshot
    ):
        alpha_opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=10.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        beta_opened = create_execution(
            execution_id="e2",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T1,
            effect="open",
            tag="1h-b",
        )
        stop_loss = create_execution(
            execution_id="e3",
            side=OrderSide.SELL,
            quantity=4.0,
            timestamp=T2,
            effect="close",
            kind=OrderType.STOP_LOSS,
        )

        tracker.refresh(
            create_snapshot(
                [alpha_opened, beta_opened, stop_loss], venue_quantity=10.0
            ),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 10.0)
        assert tracker.get(BETA) is None

    def test_a_fill_no_rule_settles_is_warned_as_opened_outside(
        self, tracker, create_execution, create_snapshot, caplog
    ):
        alpha_opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=10.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        beta_opened = create_execution(
            execution_id="e2",
            side=OrderSide.BUY,
            quantity=8.0,
            timestamp=T1,
            effect="open",
            tag="1h-b",
        )
        stop_loss = create_execution(
            execution_id="e3",
            side=OrderSide.SELL,
            quantity=4.0,
            timestamp=T2,
            effect="close",
            kind=OrderType.STOP_LOSS,
        )

        with caplog.at_level(logging.INFO):
            tracker.refresh(
                create_snapshot(
                    [alpha_opened, beta_opened, stop_loss], venue_quantity=14.0
                ),
                CANDLE_OPEN,
                CANDLE_CLOSE,
            )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 10.0)
        assert tracker.get(BETA) == TrackedPosition(PositionSide.LONG, 8.0)
        assert [record.levelname for record in caplog.records] == ["WARNING"]
        assert "-4" in caplog.text

    def test_an_unlabelled_fill_of_another_kind_stays_outside(
        self, tracker, create_execution, create_snapshot, opened_by_alpha, caplog
    ):
        by_hand = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=4.0,
            timestamp=T1,
            effect="close",
            kind=OrderType.MARKET,
        )

        with caplog.at_level(logging.INFO):
            tracker.refresh(
                create_snapshot([opened_by_alpha, by_hand], venue_quantity=6.0),
                CANDLE_OPEN,
                CANDLE_CLOSE,
            )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 10.0)
        assert [record.levelname for record in caplog.records] == ["WARNING"]

    def test_a_protection_carrying_another_bots_tag_stays_outside(
        self, tracker, create_execution, create_snapshot, opened_by_alpha
    ):
        stop_loss = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=4.0,
            timestamp=T1,
            effect="close",
            kind=OrderType.STOP_LOSS,
            tag="1h-other",
        )

        tracker.refresh(
            create_snapshot([opened_by_alpha, stop_loss], venue_quantity=6.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 10.0)

    def test_a_protection_taking_the_venue_through_flat_still_clears_every_id(
        self, tracker, create_execution, create_snapshot, opened_by_alpha
    ):
        stop_loss = create_execution(
            execution_id="e2",
            side=OrderSide.SELL,
            quantity=10.0,
            timestamp=T1,
            effect="close",
            kind=OrderType.STOP_LOSS,
        )

        tracker.refresh(
            create_snapshot([opened_by_alpha, stop_loss], venue_quantity=None),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) is None

    def test_a_stop_closing_more_than_the_only_long_holds_is_not_credited_to_it(
        self, tracker, create_execution, create_snapshot
    ):
        alpha_opened = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=4.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        by_hand = create_execution(
            execution_id="e2",
            side=OrderSide.BUY,
            quantity=6.0,
            timestamp=T1,
            effect="open",
            kind=OrderType.MARKET,
        )
        stop_loss = create_execution(
            execution_id="e3",
            side=OrderSide.SELL,
            quantity=6.0,
            timestamp=T2,
            effect="close",
            kind=OrderType.STOP_LOSS,
        )

        tracker.refresh(
            create_snapshot([alpha_opened, by_hand, stop_loss], venue_quantity=4.0),
            CANDLE_OPEN,
            CANDLE_CLOSE,
        )

        assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 4.0)


class TestMultipleSymbols:
    def test_each_symbol_is_folded_independently(self, create_execution):
        eth = Symbol.create("ETH/USDT:USDT")
        gamma = TrackingId("ETH/USDT:USDT@1d#gamma")
        tracker = ExchangePositionTracker.create(
            {BTC: (ALPHA,), eth: (gamma,)},
            {ALPHA: profile_tag("1h", "a"), gamma: profile_tag("1h", "g")},
        )
        btc_fill = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=3.0,
            timestamp=T0,
            effect="open",
            tag="1h-a",
        )
        eth_fill = create_execution(
            execution_id="e2",
            side=OrderSide.BUY,
            quantity=5.0,
            timestamp=T0,
            effect="open",
            tag="1h-g",
            symbol=eth,
        )
        snapshot = AccountSnapshot(
            account_name="test",
            positions={
                BTC: PositionSnapshot(
                    symbol=BTC,
                    side=PositionSide.LONG,
                    quantity=3.0,
                    average_entry_price=100.0,
                    entry_time=T0,
                    leverage=1.0,
                    liquidation_price=0.0,
                    stop_loss_price=None,
                    take_profit_price=None,
                ),
                eth: PositionSnapshot(
                    symbol=eth,
                    side=PositionSide.LONG,
                    quantity=5.0,
                    average_entry_price=100.0,
                    entry_time=T0,
                    leverage=1.0,
                    liquidation_price=0.0,
                    stop_loss_price=None,
                    take_profit_price=None,
                ),
            },
            executions=[btc_fill, eth_fill],
            executions_declared=True,
        )

        tracker.refresh(snapshot, CANDLE_OPEN, CANDLE_CLOSE)

        assert tracker.get(ALPHA).quantity == 3.0
        assert tracker.get(gamma).quantity == 5.0

    def test_a_tag_declared_for_another_symbol_is_not_credited(
        self, create_execution, create_snapshot
    ):
        eth = Symbol.create("ETH/USDT:USDT")
        gamma = TrackingId("ETH/USDT:USDT@1d#gamma")
        tracker = ExchangePositionTracker.create(
            {BTC: (ALPHA,), eth: (gamma,)},
            {ALPHA: profile_tag("1h", "a"), gamma: profile_tag("1h", "g")},
        )
        mistagged = create_execution(
            execution_id="e1",
            side=OrderSide.BUY,
            quantity=3.0,
            timestamp=T0,
            effect="open",
            tag="1h-g",
        )

        tracker.refresh(
            create_snapshot([mistagged], venue_quantity=3.0), CANDLE_OPEN, CANDLE_CLOSE
        )

        assert tracker.get(ALPHA) is None
        assert tracker.get(gamma) is None


def _tracker(*, netted: bool = False) -> ExchangePositionTracker:
    return ExchangePositionTracker.create(
        {BTC: (ALPHA, BETA)},
        {ALPHA: profile_tag("1h", "a"), BETA: profile_tag("1h", "b")},
        netted=netted,
    )
