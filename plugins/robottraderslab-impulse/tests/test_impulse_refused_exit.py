from collections.abc import Awaitable, Callable
from typing import Any

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
from robottraderslab.exceptions import ExchangeRecoverableError
from robottraderslab.exchanges import OrderSide, PlacedOrder

VENUE_ANSWER = "25202 Insufficient balance"
HOLDER_PROFILE: dict[str, Any] = {**BTC_PROFILE, "tag": HOLDER_TAG}


class TestAnExitTheVenueRefused:
    @pytest.fixture
    async def retried_exit(self, run_cycle: Callable[..., Awaitable[None]]) -> None:
        await run_cycle()
        await run_cycle(refusing=False)

    @pytest.mark.usefixtures("retried_exit")
    async def test_a_flat_position_books_no_exit_on_the_cycle_after(
        self, refusing_exchange, run_cycle
    ):
        fills_after_the_exit = executions_of(refusing_exchange)

        await run_cycle(refusing=False)

        assert executions_of(refusing_exchange) == fills_after_the_exit

    async def test_a_side_open_again_books_no_exit_and_the_venue_holds_what_the_strategy_holds(
        self, refusing_exchange, run_cycle
    ):
        await run_cycle()

        await run_cycle(side_closed=False)

        assert refusing_exchange.refusals == 1
        assert refusing_exchange.simulation_engine.open_positions.keys() == {BTC}
        assert executed_sides_and_quantities(refusing_exchange) == [
            (OrderSide.BUY, HELD_QUANTITY)
        ]

    @pytest.mark.usefixtures("retried_exit")
    async def test_no_stop_loss_rests_once_the_exit_has_filled(self, refusing_exchange):
        assert await resting_stop_losses(refusing_exchange) == []

    @pytest.mark.usefixtures("retried_exit")
    async def test_the_next_cycle_places_the_exit_again_and_the_position_is_flat(
        self, refusing_exchange
    ):
        assert refusing_exchange.simulation_engine.open_positions == {}
        assert executed_sides_and_quantities(refusing_exchange) == [
            (OrderSide.BUY, HELD_QUANTITY),
            (OrderSide.SELL, HELD_QUANTITY),
        ]

    async def test_the_refusal_is_logged_at_error_naming_the_symbol(
        self, run_cycle, caplog
    ):
        await run_cycle()

        refusal_lines = [
            log_record
            for log_record in caplog.records
            if VENUE_ANSWER in log_record.getMessage()
        ]
        assert [log_record.levelname for log_record in refusal_lines] == ["ERROR"]
        assert str(BTC) in refusal_lines[0].getMessage()

    async def test_the_stop_loss_keeps_guarding_the_position_after_the_refused_cycle(
        self, refusing_exchange, run_cycle
    ):
        await run_cycle()

        assert len(await resting_stop_losses(refusing_exchange)) == 1


class _RefusingExchange(SimulatedFuturesExchange):
    _refusing: bool = False
    _refusals: int = 0

    @property
    def refusals(self) -> int:
        return self._refusals

    async def place_market_order(
        self, symbol: Symbol, side: OrderSide, *args: Any, **kwargs: Any
    ) -> PlacedOrder:
        if self._refusing and side is OrderSide.SELL:
            self._refusals += 1
            raise ExchangeRecoverableError(VENUE_ANSWER)
        return await super().place_market_order(symbol, side, *args, **kwargs)

    def set_refusing(self, refusing: bool) -> None:
        self._refusing = refusing


@pytest.fixture
async def refusing_exchange() -> _RefusingExchange:
    held_exchange = await hold_a_long_with_a_stop(_RefusingExchange, HOLDER_TAG)
    held_exchange.set_refusing(True)
    (_held_stop_loss,) = await resting_stop_losses(held_exchange)
    return held_exchange


@pytest.fixture
def run_cycle(
    refusing_exchange: _RefusingExchange,
) -> Callable[..., Awaitable[None]]:
    """Run one cycle on a strategy and a bot built afresh, as its own process would.

    The venue refuses the exit unless `refusing` is False, and the profile's side
    is closed on the last candle unless `side_closed` is False.
    """
    closed_side_impulse = scripted_impulse({HOLDER_TAG: {LAST_CANDLE: "long_exit"}})
    open_side_impulse = scripted_impulse({HOLDER_TAG: {}})

    async def run(*, refusing: bool = True, side_closed: bool = True) -> None:
        refusing_exchange.set_refusing(refusing)
        strategy_class = closed_side_impulse if side_closed else open_side_impulse
        await run_one_cycle(refusing_exchange, strategy_class, [HOLDER_PROFILE])

    return run
