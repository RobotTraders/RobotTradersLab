import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import time_machine
from dry_run_helpers import (
    BTC,
    CANCELLED_ORDER,
    ETH,
    HELD_SETTINGS,
    MESSAGE,
    BookedCall,
    PlainStrategy,
    btc_and_eth_probe,
    btc_probe,
    dry_run,
    dry_run_at_debug,
    dry_run_probe,
    lines_a_failure_must_not_log,
    probe,
    section,
    sent,
)
from live_cycle_helpers import (
    bought,
    livebot,
    make_provider,
    messages_at,
    requirements_failing_to_refresh,
)
from readouts import outcomes

from robottraderslab._internal import DryRunFailure
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
    SymbolNotListedError,
)
from robottraderslab.exchanges import PlacedOrder
from robottraderslab.live.livebot import LiveBot
from robottraderslab.strategies import StrategyProtocol, StrategyRequirements

MIDNIGHT = datetime(2026, 1, 1, tzinfo=UTC)
NOON = datetime(2026, 1, 1, 12, tzinfo=UTC)
A_MINUTE_PAST_NOON = datetime(2026, 1, 1, 12, 1, tzinfo=UTC)
A_MINUTE_PAST_MIDNIGHT = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)

DECLARED_LEVERAGE = 3.0


async def dry_run_logging_at_debug_at_most(
    caplog: pytest.LogCaptureFixture, strategy: StrategyProtocol, provider: Mock
) -> list[DryRunFailure]:
    """Dry run one cycle, return what it met and check it logged no unwanted line."""
    failures = await dry_run_at_debug(caplog, strategy, provider)
    assert lines_a_failure_must_not_log(caplog) == []
    return failures


def midnight_livebot(strategy: StrategyProtocol, **options: Any) -> LiveBot:
    """A live cycle of the strategy whose moment is midnight."""
    return livebot(strategy, clock=lambda: MIDNIGHT, **options)


async def _notify(notification: Any) -> None:
    """What a subscriber is handed."""


@pytest.fixture
def exchange(make_live_venue: Callable[[], Mock]) -> Mock:
    venue = make_live_venue()
    venue.get_margin_settings.side_effect = lambda *_: {
        BTC: HELD_SETTINGS,
        ETH: HELD_SETTINGS,
    }
    return venue


@pytest.fixture
def ohlcv_provider() -> Mock:
    return make_provider()


class TestWhatItSends:
    async def test_a_dry_run_sends_and_notifies_nothing_where_a_cycle_does(
        self, account, exchange, ohlcv_provider
    ):
        exchange.place_market_order.return_value = PlacedOrder(
            order_id="order-1", get_fill=bought
        )
        on_placement = AsyncMock(spec=_notify)
        on_fill = AsyncMock(spec=_notify)
        sections = [section(BTC, sends_everything=True)]

        failures = await dry_run_probe(
            account,
            sections,
            ohlcv_provider,
            on_placement=[on_placement],
            on_fill=[on_fill],
        )

        assert failures == []
        assert sent(exchange) == []
        on_placement.assert_not_awaited()
        on_fill.assert_not_awaited()

        await midnight_livebot(
            probe(account, *sections),
            on_placement=[on_placement],
            on_fill=[on_fill],
        ).run()

        assert sent(exchange) == [
            "cancel_order_by_id",
            "place_market_order",
            "set_leverage",
            "set_margin_mode",
        ]
        on_placement.assert_awaited_once()
        on_fill.assert_awaited_once()
        exchange.cancel_order_by_id.assert_awaited_once_with(BTC, CANCELLED_ORDER)

    async def test_a_declared_leverage_is_booked_on_and_never_set_on_the_venue(
        self, account, exchange, ohlcv_provider
    ):
        sections = [section(BTC, leverage=DECLARED_LEVERAGE)]
        dry_strategy = probe(account, *sections)

        await dry_run(dry_strategy, ohlcv_provider)

        exchange.set_leverage.assert_not_awaited()
        exchange.set_margin_mode.assert_not_awaited()
        [dry_booking] = dry_strategy.booked
        assert dry_booking.quantity == DECLARED_LEVERAGE

        cycle_strategy = probe(account, *sections)
        await midnight_livebot(cycle_strategy).run()

        exchange.set_leverage.assert_awaited_once_with(BTC, DECLARED_LEVERAGE)
        [placed_order] = exchange.place_market_order.await_args_list
        assert placed_order.kwargs["quantity"] == dry_booking.quantity


class TestWhatItRuns:
    @pytest.mark.parametrize("wall_clock", [NOON, A_MINUTE_PAST_NOON])
    async def test_every_slot_runs_once_whether_a_declared_candle_closes_or_not(
        self, account, ohlcv_provider, wall_clock
    ):
        strategy = probe(account, section(BTC, "1h"), section(BTC, "1d"))

        failures = await dry_run(strategy, ohlcv_provider, clock=lambda: wall_clock)

        assert failures == []
        assert strategy.phases == [
            "setup",
            "general_signals",
            "general_booking",
            "final_booking",
        ]
        assert strategy.signalled == [BTC, BTC]
        assert [(call.symbol, call.timeframe) for call in strategy.booked] == [
            (BTC, "1h"),
            (BTC, "1d"),
        ]

    async def test_a_strategy_that_is_no_profile_strategy_runs_once_on_every_timeframe(
        self, account, ohlcv_provider
    ):
        strategy = PlainStrategy(account)

        failures = await dry_run(strategy, ohlcv_provider)

        assert failures == []
        assert strategy.signal_calls == 1
        assert strategy.booking_timeframes == [["1h", "1d", "1w"]]


class TestTheCyclesMoment:
    async def test_a_cycle_books_the_candles_closed_at_its_clock_not_at_the_wall_clock(
        self, account
    ):
        strategy = btc_probe(account)

        with time_machine.travel(A_MINUTE_PAST_MIDNIGHT, tick=False):
            await midnight_livebot(strategy).run()

        assert strategy.booked == [BookedCall(BTC, "1d", 1.0)]

    @pytest.mark.usefixtures("candle_boundary_clock")
    async def test_a_cycle_without_a_clock_books_the_candles_closed_at_the_wall_clock(
        self, account
    ):
        strategy = btc_probe(account)

        await livebot(strategy).run()

        assert strategy.booked == [BookedCall(BTC, "1d", 1.0)]


class TestADroppedTimeframe:
    async def test_a_cycle_books_the_closing_timeframes_the_fetch_holds_and_not_the_others(
        self, account
    ):
        strategy = probe(account, section(BTC, "1h"), section(ETH, "1d"))
        provider = make_provider({ETH: ExchangeRecoverableError(MESSAGE)})

        await LiveBot(strategy, provider, clock=lambda: MIDNIGHT).run()

        assert strategy.booked == [BookedCall(BTC, "1h", 1.0)]


class TestWhatItReturns:
    @pytest.mark.parametrize(
        "btc_fields",
        [
            {"refused": True},
            {"fail": "signals:ValueError"},
            {"fail": "signals:BotCriticalError"},
            {"fail": "booking:ValueError"},
            {"fail": "booking:BotCriticalError"},
        ],
        ids=[
            "a refused section",
            "a signal slot raising",
            "a signal slot raising a critical error of the bot",
            "a booking slot raising",
            "a booking slot raising a critical error of the bot",
        ],
    )
    async def test_a_profile_failure_names_its_symbol_and_the_other_profiles_run(
        self, account, ohlcv_provider, btc_fields
    ):
        strategy = btc_and_eth_probe(account, **btc_fields)

        failures = await dry_run(strategy, ohlcv_provider)

        assert outcomes(failures) == [(BTC, False)]
        assert ETH in [call.symbol for call in strategy.booked]

    @pytest.mark.parametrize(
        ("fail", "recoverable"),
        [
            ("setup:ValueError", False),
            ("setup:BotCriticalError", False),
            ("general_signals:ValueError", False),
            ("general_signals:ExchangeRecoverableError", True),
            ("general_booking:ValueError", False),
            ("general_booking:ExchangeRecoverableError", True),
            ("final_booking:ValueError", False),
        ],
        ids=[
            "setup",
            "setup raising a critical error of the bot",
            "a general signal slot",
            "a general signal slot of the recoverable family",
            "a general booking slot",
            "a general booking slot of the recoverable family",
            "a final slot",
        ],
    )
    async def test_a_failure_outside_a_profile_names_no_symbol(
        self, account, ohlcv_provider, fail, recoverable
    ):
        failures = await dry_run_probe(
            account, [section(BTC, fail=fail)], ohlcv_provider
        )

        assert outcomes(failures) == [(None, recoverable)]

    @pytest.mark.parametrize(
        ("error", "recoverable"),
        [
            ("ExchangeRecoverableError", True),
            ("ExchangeTransientError", True),
            ("MissingOhlcvDataError", True),
            ("ValueError", False),
            ("ProfileError", False),
            ("BotCriticalError", False),
        ],
    )
    async def test_only_the_recoverable_family_is_marked_recoverable(
        self, account, ohlcv_provider, error, recoverable
    ):
        failures = await dry_run_probe(
            account, [section(BTC, fail=f"booking:{error}")], ohlcv_provider
        )

        assert outcomes(failures) == [(BTC, recoverable)]

    @pytest.mark.parametrize(
        ("error", "sentence"),
        [
            (ExchangeRecoverableError(MESSAGE), f"{ETH}@1d: skipping this cycle: boom"),
            (
                SymbolNotListedError(MESSAGE),
                f"{ETH}@1d: its profiles sit out this cycle: boom",
            ),
        ],
        ids=["a venue that did not answer", "a symbol the venue does not list"],
    )
    async def test_a_pair_the_fetch_dropped_is_a_recoverable_failure_of_its_symbol(
        self, account, error, sentence
    ):
        strategy = btc_and_eth_probe(account)

        failures = await dry_run(strategy, make_provider({ETH: error}))

        assert failures == [DryRunFailure(ETH, sentence, recoverable=True)]
        assert BTC in [call.symbol for call in strategy.booked]

    @pytest.mark.parametrize("slot", ["signals", "booking"])
    async def test_a_profile_slot_failure_says_what_a_cycle_logs_of_it(
        self, account, ohlcv_provider, caplog, slot
    ):
        fail = f"{slot}:ValueError"
        dry_strategy = btc_probe(account, fail=fail)
        cycle = midnight_livebot(btc_probe(account, fail=fail))
        [profile] = dry_strategy.profiles

        [failure] = await dry_run(dry_strategy, ohlcv_provider)
        with caplog.at_level(logging.ERROR):
            await cycle.run()

        assert profile.profile_id in failure.message
        assert f"the {slot} slot raised ValueError: {MESSAGE}" in failure.message
        assert messages_at(caplog, logging.ERROR) == [failure.message]
        assert cycle.failures == []

    async def test_candles_none_of_which_could_be_fetched_end_the_run_recoverably(
        self, account
    ):
        provider = make_provider(
            {BTC: ExchangeRecoverableError(MESSAGE), ETH: SymbolNotListedError(MESSAGE)}
        )
        strategy = btc_and_eth_probe(account)

        failures = await dry_run(strategy, provider)

        assert outcomes(failures) == [(BTC, True), (ETH, True), (None, True)]
        assert "No declared market data could be fetched" in failures[-1].message
        assert strategy.booked == []


class TestAReadThatFails:
    async def test_a_state_read_the_venue_did_not_answer_ends_the_cycle_before_booking(
        self, account, exchange, ohlcv_provider, caplog
    ):
        exchange.get_account_state.side_effect = ExchangeRecoverableError(MESSAGE)
        dry_strategy = btc_probe(account)
        cycle_strategy = btc_probe(account)

        failures = await dry_run(dry_strategy, ohlcv_provider)
        with caplog.at_level(logging.WARNING):
            await midnight_livebot(cycle_strategy).run()

        assert outcomes(failures) == [(None, True)]
        assert dry_strategy.booked == []
        assert messages_at(caplog, logging.WARNING) == [failures[0].message]
        assert "Account state fetch failed, skipping cycle" in failures[0].message
        assert cycle_strategy.booked == []
        assert sent(exchange) == []

    @pytest.mark.parametrize(
        ("error", "level", "sentence", "recoverable"),
        [
            (
                ExchangeRecoverableError(MESSAGE),
                logging.WARNING,
                f"Tracked position settlement failed, skipping cycle: {MESSAGE}",
                True,
            ),
            (
                ValueError(MESSAGE),
                logging.ERROR,
                f"Internal error refreshing tracked positions: {MESSAGE}",
                False,
            ),
        ],
        ids=["a settlement the venue did not answer", "an internal error"],
    )
    async def test_a_tracked_position_refresh_that_fails_ends_the_cycle_before_booking(
        self,
        account,
        exchange,
        ohlcv_provider,
        caplog,
        error,
        level,
        sentence,
        recoverable,
    ):
        def refuse() -> StrategyRequirements:
            return requirements_failing_to_refresh(error)

        dry_strategy = btc_probe(account)
        cycle_strategy = btc_probe(account)

        failures = await dry_run(
            dry_strategy, ohlcv_provider, requirements_factory=refuse
        )
        with caplog.at_level(logging.WARNING):
            await midnight_livebot(cycle_strategy, requirements_factory=refuse).run()

        assert outcomes(failures) == [(None, recoverable)]
        assert failures[0].message == sentence
        assert dry_strategy.booked == []
        assert messages_at(caplog, level) == [sentence]
        assert cycle_strategy.booked == []
        assert sent(exchange) == []


class TestWhatItRaises:
    @pytest.mark.parametrize(
        "fail",
        ["setup:ExchangeCriticalError", "signals:ExchangeCriticalError"]
        + ["general_booking:ExchangeCriticalError", "booking:ExchangeCriticalError"],
    )
    async def test_an_exchange_critical_error_reaches_the_caller_from_any_step(
        self, account, ohlcv_provider, fail
    ):
        with pytest.raises(ExchangeCriticalError):
            await dry_run_probe(account, [section(BTC, fail=fail)], ohlcv_provider)


class TestWhatItLogs:
    @pytest.mark.parametrize(
        "btc_fields",
        [
            {"fail": "setup:BotCriticalError"},
            {"fail": "general_signals:ValueError"},
            {"fail": "signals:ValueError"},
            {"fail": "signals:BotCriticalError"},
            {"fail": "booking:ValueError"},
            {"fail": "general_booking:ExchangeRecoverableError"},
            {"fail": "final_booking:ValueError"},
            {"refused": True},
        ],
    )
    async def test_a_failure_it_returns_is_logged_at_debug_at_most(
        self, account, ohlcv_provider, caplog, btc_fields
    ):
        strategy = btc_and_eth_probe(account, **btc_fields)

        failures = await dry_run_logging_at_debug_at_most(
            caplog, strategy, ohlcv_provider
        )

        assert failures

    @pytest.mark.parametrize(
        "dropped",
        [[ETH], [BTC, ETH]],
        ids=["one pair", "every pair"],
    )
    async def test_a_dropped_pair_is_logged_at_debug_at_most(
        self, account, caplog, dropped
    ):
        provider = make_provider(
            {symbol: ExchangeRecoverableError(MESSAGE) for symbol in dropped}
        )
        strategy = btc_and_eth_probe(account)

        failures = await dry_run_logging_at_debug_at_most(caplog, strategy, provider)

        assert len(failures) >= len(dropped)
