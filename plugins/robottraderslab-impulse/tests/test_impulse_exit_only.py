from collections.abc import Awaitable, Callable

import pytest
from impulse_helpers import (
    BTC,
    BTC_PROFILE,
    HELD_QUANTITY,
    HOLDER_TAG,
    LAST_CANDLE,
    executed_sides_and_quantities,
    executions_of,
    hold_a_long_with_a_stop,
    resting_stop_losses,
    run_one_cycle,
    scripted_impulse,
)

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import SimulatedFuturesExchange
from robottraderslab.exchanges import OrderSide

ENTERING = ("alpha", "beta")
PROFILES = [{**BTC_PROFILE, "tag": tag} for tag in (*ENTERING, HOLDER_TAG)]
SIGNALS_BY_TAG = {
    "alpha": {LAST_CANDLE: "long_entry"},
    "beta": {LAST_CANDLE: "long_entry"},
    HOLDER_TAG: {LAST_CANDLE: "long_exit"},
}


class _CancelRecordingExchange(SimulatedFuturesExchange):
    cancelled_ids: list[str]

    async def cancel_order_by_id(self, symbol: Symbol, order_id: str) -> None:
        self.cancelled_ids.append(order_id)
        await super().cancel_order_by_id(symbol, order_id)


@pytest.fixture
async def cancel_recording_exchange() -> _CancelRecordingExchange:
    held_exchange = await hold_a_long_with_a_stop(_CancelRecordingExchange, HOLDER_TAG)
    held_exchange.cancelled_ids = []
    (_held_stop_loss,) = await resting_stop_losses(held_exchange)
    return held_exchange


@pytest.fixture
def run_cycle(
    cancel_recording_exchange: _CancelRecordingExchange,
) -> Callable[[frozenset[Symbol]], Awaitable[None]]:
    async def run(exit_only_symbols: frozenset[Symbol]) -> None:
        await run_one_cycle(
            cancel_recording_exchange,
            scripted_impulse(SIGNALS_BY_TAG),
            PROFILES,
            exit_only_symbols=exit_only_symbols,
        )

    return run


class TestTheLastProfileOfAStackedSymbolExits:
    @pytest.fixture
    async def exit_only_cycle(
        self, run_cycle: Callable[[frozenset[Symbol]], Awaitable[None]]
    ) -> None:
        await run_cycle(frozenset({BTC}))

    @pytest.mark.usefixtures("exit_only_cycle")
    async def test_the_cycle_books_no_cancel_of_the_stop_loss(
        self, cancel_recording_exchange
    ):
        assert cancel_recording_exchange.cancelled_ids == []

    async def test_the_entries_of_the_other_profiles_are_sent_on_a_symbol_running_as_usual(
        self, cancel_recording_exchange, run_cycle
    ):
        await run_cycle(frozenset())

        buy_count = [
            execution.side for execution in executions_of(cancel_recording_exchange)
        ].count(OrderSide.BUY)
        assert buy_count == 3

    @pytest.mark.usefixtures("exit_only_cycle")
    async def test_the_exit_is_the_only_order_the_cycle_sent(
        self, cancel_recording_exchange
    ):
        assert executed_sides_and_quantities(cancel_recording_exchange) == [
            (OrderSide.BUY, HELD_QUANTITY),
            (OrderSide.SELL, HELD_QUANTITY),
        ]
