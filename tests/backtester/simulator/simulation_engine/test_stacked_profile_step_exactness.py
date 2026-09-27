import logging
from datetime import datetime

import pytest

from robottraderslab._core import (
    AccountSnapshot,
    PlacementReserve,
    execute_trading_actions,
)
from robottraderslab._core.exchange_position_tracker import ExchangePositionTracker
from robottraderslab.backtester.simulator import FeeModel, fee_model_for
from robottraderslab.strategies import (
    BookKeeper,
    PositionSide,
    TrackedPosition,
    TrackingId,
    profile_tag,
)
from robottraderslab.strategies.futures import FuturesAccount

_NO_RESERVE = PlacementReserve(margin_markup=0.0, fee_markup=0.0, notional_reserve=0.0)
_PRICE = 100.0
_SINCE = datetime.min
_UNTIL = datetime.max

_ALPHA_TAG = "1h-a"
_BETA_TAG = "1h-b"
ALPHA = TrackingId("BTC/USDT:USDT@1h-a")
BETA = TrackingId("BTC/USDT:USDT@1h-b")

_BETA_ENTRY_QUANTITY = 0.00005
_ALPHA_ENTRY_QUANTITY = 0.00001037
_ALPHA_PARTIAL_EXIT_QUANTITY = 0.00001036
_ALPHA_RESIDUAL_QUANTITY = 1e-8
_ALPHA_RE_ENTRY_QUANTITY = 0.0002

_ALPHA_SHORT_QUANTITY = 0.00002071
_ALPHA_SECOND_LONG_QUANTITY = 0.00001553
_BETA_SHORT_QUANTITY = 0.00003331
_BETA_SECOND_LONG_QUANTITY = 0.000042


@pytest.fixture
def fee_model() -> FeeModel:
    return fee_model_for("exchange", _NO_RESERVE)


async def _book(*actions) -> None:
    bookkeeper = BookKeeper()
    for action in actions:
        bookkeeper.add(action)
    await execute_trading_actions(bookkeeper.list_actions())


def _refresh(tracker, sim) -> None:
    tracker.refresh(
        AccountSnapshot(
            account_name="test",
            positions=sim.simulation_engine.open_positions,
            executions=sim.simulation_engine.get_executions_since(_SINCE),
            executions_declared=True,
        ),
        _SINCE,
        _UNTIL,
    )


def _tracker(btc_usdt_perp) -> ExchangePositionTracker:
    return ExchangePositionTracker.create(
        {btc_usdt_perp: (ALPHA, BETA)},
        {ALPHA: profile_tag("1h", "a"), BETA: profile_tag("1h", "b")},
    )


async def _stack_and_partially_exit_alpha(sim, btc_usdt_perp):
    """Beta holds the symbol throughout; alpha enters, then partially exits by
    one step less than it entered, in quantities whose float difference lands
    one ulp under the whole step it should leave behind.
    """
    account = FuturesAccount(sim.exchange)
    tracker = _tracker(btc_usdt_perp)
    await _book(
        account.long_entry(btc_usdt_perp, _BETA_ENTRY_QUANTITY).tag(_BETA_TAG).build(),
    )
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=_PRICE)
    await _book(
        account.long_entry(btc_usdt_perp, _ALPHA_ENTRY_QUANTITY)
        .tag(_ALPHA_TAG)
        .build(),
    )
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=_PRICE)
    await _book(
        account.long_exit(btc_usdt_perp, _ALPHA_PARTIAL_EXIT_QUANTITY, reduce_only=True)
        .tag(_ALPHA_TAG)
        .build(),
    )
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=_PRICE)
    _refresh(tracker, sim)
    return account, tracker, tracker.get(ALPHA)


class TestClosingTheLastStepAStackedProfileHolds:
    async def test_a_partial_exit_leaves_each_profile_holding_only_its_own_fills(
        self, sim, btc_usdt_perp
    ):
        _, tracker, residual = await _stack_and_partially_exit_alpha(sim, btc_usdt_perp)

        assert residual == TrackedPosition(PositionSide.LONG, _ALPHA_RESIDUAL_QUANTITY)
        assert tracker.get(BETA) == TrackedPosition(
            PositionSide.LONG, _BETA_ENTRY_QUANTITY
        )

    async def test_closing_the_residual_is_never_refused_for_holding_under_one_step(
        self, sim, btc_usdt_perp, caplog
    ):
        account, tracker, residual = await _stack_and_partially_exit_alpha(
            sim, btc_usdt_perp
        )

        with caplog.at_level(logging.WARNING):
            await _book(
                account.long_exit(btc_usdt_perp, residual.quantity, reduce_only=False)
                .tag(_ALPHA_TAG)
                .build(),
            )
            sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=_PRICE)

        assert "under one quantity step" not in caplog.text
        _refresh(tracker, sim)
        assert tracker.get(ALPHA) is None
        assert tracker.get(BETA) == TrackedPosition(
            PositionSide.LONG, _BETA_ENTRY_QUANTITY
        )

    async def test_a_profile_that_closed_its_residual_takes_its_next_entry(
        self, sim, btc_usdt_perp
    ):
        account, tracker, residual = await _stack_and_partially_exit_alpha(
            sim, btc_usdt_perp
        )
        await _book(
            account.long_exit(btc_usdt_perp, residual.quantity, reduce_only=False)
            .tag(_ALPHA_TAG)
            .build(),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=_PRICE)

        await _book(
            account.long_entry(btc_usdt_perp, _ALPHA_RE_ENTRY_QUANTITY)
            .tag(_ALPHA_TAG)
            .build(),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=_PRICE)
        _refresh(tracker, sim)

        assert tracker.get(ALPHA) == TrackedPosition(
            PositionSide.LONG, _ALPHA_RE_ENTRY_QUANTITY
        )


class TestTwoProfilesFlippingOnOneSymbol:
    async def test_each_profile_holds_only_what_its_own_fills_add_up_to(
        self, sim, btc_usdt_perp
    ):
        account = FuturesAccount(sim.exchange)
        tracker = _tracker(btc_usdt_perp)

        await _book(
            account.long_entry(btc_usdt_perp, _ALPHA_ENTRY_QUANTITY)
            .tag(_ALPHA_TAG)
            .build(),
            account.long_entry(btc_usdt_perp, _BETA_ENTRY_QUANTITY)
            .tag(_BETA_TAG)
            .build(),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=_PRICE)

        await _book(
            account.long_exit(btc_usdt_perp, _ALPHA_ENTRY_QUANTITY, reduce_only=False)
            .tag(_ALPHA_TAG)
            .build(),
            account.short_entry(btc_usdt_perp, _ALPHA_SHORT_QUANTITY)
            .tag(_ALPHA_TAG)
            .build(),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=_PRICE)
        _refresh(tracker, sim)

        assert tracker.get(ALPHA) == TrackedPosition(
            PositionSide.SHORT, _ALPHA_SHORT_QUANTITY
        )
        assert tracker.get(BETA) == TrackedPosition(
            PositionSide.LONG, _BETA_ENTRY_QUANTITY
        )

        await _book(
            account.long_exit(btc_usdt_perp, _BETA_ENTRY_QUANTITY, reduce_only=False)
            .tag(_BETA_TAG)
            .build(),
            account.short_entry(btc_usdt_perp, _BETA_SHORT_QUANTITY)
            .tag(_BETA_TAG)
            .build(),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=_PRICE)
        _refresh(tracker, sim)

        assert tracker.get(ALPHA) == TrackedPosition(
            PositionSide.SHORT, _ALPHA_SHORT_QUANTITY
        )
        assert tracker.get(BETA) == TrackedPosition(
            PositionSide.SHORT, _BETA_SHORT_QUANTITY
        )

        await _book(
            account.short_exit(btc_usdt_perp, _ALPHA_SHORT_QUANTITY, reduce_only=False)
            .tag(_ALPHA_TAG)
            .build(),
            account.long_entry(btc_usdt_perp, _ALPHA_SECOND_LONG_QUANTITY)
            .tag(_ALPHA_TAG)
            .build(),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=_PRICE)

        await _book(
            account.short_exit(btc_usdt_perp, _BETA_SHORT_QUANTITY, reduce_only=False)
            .tag(_BETA_TAG)
            .build(),
            account.long_entry(btc_usdt_perp, _BETA_SECOND_LONG_QUANTITY)
            .tag(_BETA_TAG)
            .build(),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=_PRICE)
        _refresh(tracker, sim)

        assert tracker.get(ALPHA) == TrackedPosition(
            PositionSide.LONG, _ALPHA_SECOND_LONG_QUANTITY
        )
        assert tracker.get(BETA) == TrackedPosition(
            PositionSide.LONG, _BETA_SECOND_LONG_QUANTITY
        )
