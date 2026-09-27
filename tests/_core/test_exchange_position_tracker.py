import logging
from collections.abc import Callable
from datetime import datetime, timezone

import pytest

from robottraderslab import Symbol
from robottraderslab._core.actions.members.tags import profile_tag
from robottraderslab._core.exchange_position_tracker import ExchangePositionTracker
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.exchanges import (
    FillEffect,
    PositionSide,
    client_order_id_carrying,
)
from robottraderslab.strategies import (
    AccountSnapshot,
    Execution,
    OrderSide,
    TrackedPosition,
    TrackingId,
)
from robottraderslab.strategies.futures import PositionSnapshot

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
        with pytest.raises(StrategyCriticalError, match="have no tag"):
            ExchangePositionTracker.create(
                {BTC: (ALPHA, BETA)},
                {ALPHA: profile_tag("1h", "a")},
            )

    def test_two_ids_sharing_a_tag_on_one_symbol_are_refused(self):
        with pytest.raises(StrategyCriticalError, match="share the tag"):
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
        """A live cycle's executions window slides forward with the candles it
        fetches, so an old fill drops out of it well before its effect on
        exposure should. Refolding the window fresh each candle would lose that
        fill's effect the moment it ages out; only a running total that
        persists across refreshes keeps it.
        """
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
                snapshot, [by_hand, opened], READ_SINCE, CANDLE_OPEN, CANDLE_CLOSE
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
            tracker.widen(snapshot, [opened], READ_SINCE, CANDLE_OPEN, CANDLE_CLOSE)

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
                ),
                eth: PositionSnapshot(
                    symbol=eth,
                    side=PositionSide.LONG,
                    quantity=5.0,
                    average_entry_price=100.0,
                    entry_time=T0,
                    leverage=1.0,
                    liquidation_price=0.0,
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


def _tracker() -> ExchangePositionTracker:
    return ExchangePositionTracker.create(
        {BTC: (ALPHA, BETA)},
        {ALPHA: profile_tag("1h", "a"), BETA: profile_tag("1h", "b")},
    )
