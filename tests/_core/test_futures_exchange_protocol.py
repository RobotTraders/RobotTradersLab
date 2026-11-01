import asyncio
from dataclasses import replace
from datetime import datetime
from typing import Any

import pytest

from robottraderslab import Symbol
from robottraderslab._core import AccountStateRequest
from robottraderslab.exceptions import ExchangeTransientError
from robottraderslab.exchanges import FuturesExchangeProtocol
from robottraderslab.futures import read_account_state


class _MarginOnlyVenue(FuturesExchangeProtocol):
    pass


def test_a_venue_reserving_nothing_locks_the_margin_alone():
    assert _MarginOnlyVenue().placement_requirement_rate(4.0) == 0.25


EXECUTIONS_SINCE = datetime(2026, 5, 1)
BTCUSDT = Symbol.create("BTC/USDT:USDT")
ETHUSDC_ON_USDT = Symbol.create("ETH/USDC:USDT")
FULL_REQUEST = AccountStateRequest(
    symbols=(BTCUSDT, ETHUSDC_ON_USDT),
    positions=True,
    open_orders=True,
    balances=True,
    margin_settings=True,
    equity=("USDT",),
    executions=True,
    base_delay=0.0,
)


class _RecordingVenue(FuturesExchangeProtocol):
    def __init__(self) -> None:
        self.events: list[str] = []
        self.failures: dict[str, list[ExchangeTransientError]] = {}

    async def _read(self, name: str, answer: Any) -> Any:
        self.events.append(f"start:{name}")
        await asyncio.sleep(0)
        failures = self.failures.get(name)
        if failures:
            raise failures.pop(0)
        self.events.append(f"answer:{name}")
        return answer

    async def get_open_orders(self, symbols):
        return await self._read("open_orders", [])

    async def get_open_positions(self, symbols):
        return await self._read("positions", {})

    async def get_balances(self, symbols):
        return await self._read("balances", {})

    async def get_quote_conversion_rate(self, symbol):
        return await self._read("rate", 1.5)

    async def get_margin_settings(self, symbols):
        return await self._read("margin_settings", {})

    async def get_equity(self, currency):
        return await self._read("equity", 7.0)

    async def get_executions_since(self, since, symbols):
        return await self._read("executions", [])

    def starts(self) -> list[str]:
        return [
            event.removeprefix("start:")
            for event in self.events
            if event.startswith("start:")
        ]


@pytest.fixture
def venue() -> _RecordingVenue:
    return _RecordingVenue()


class TestReadAccountState:
    async def test_reads_start_in_the_order_the_engine_has_always_sent_them(
        self, venue
    ):
        await read_account_state(venue, FULL_REQUEST, EXECUTIONS_SINCE)

        assert venue.starts() == [
            "open_orders",
            "positions",
            "balances",
            "rate",
            "margin_settings",
            "equity",
            "executions",
        ]

    async def test_every_read_starts_before_any_answers(self, venue):
        await read_account_state(venue, FULL_REQUEST, EXECUTIONS_SINCE)

        first_answer = next(
            index
            for index, event in enumerate(venue.events)
            if event.startswith("answer:")
        )
        independent_starts = [
            event for event in venue.events[:first_answer] if event.startswith("start:")
        ]

        assert len(independent_starts) == 6

    async def test_executions_start_once_the_positions_have_answered(self, venue):
        await read_account_state(venue, FULL_REQUEST, EXECUTIONS_SINCE)

        assert venue.events.index("answer:positions") < venue.events.index(
            "start:executions"
        )

    async def test_a_transient_failure_sends_that_read_again_and_no_other(self, venue):
        venue.failures["balances"] = [ExchangeTransientError("502")]

        await read_account_state(venue, FULL_REQUEST, EXECUTIONS_SINCE)

        assert venue.starts().count("balances") == 2
        assert venue.starts().count("open_orders") == 1
        assert venue.starts().count("positions") == 1
        assert venue.starts().count("equity") == 1
        assert venue.starts().count("executions") == 1

    async def test_a_read_failing_every_attempt_is_sent_as_often_as_the_request_allows(
        self, venue
    ):
        venue.failures["margin_settings"] = [ExchangeTransientError("502")] * 5
        request = replace(FULL_REQUEST, max_attempts=2)

        with pytest.raises(ExchangeTransientError):
            await read_account_state(venue, request, EXECUTIONS_SINCE)

        assert venue.starts().count("margin_settings") == 2

    async def test_a_symbol_quoted_outside_its_margin_without_balances_reads_no_rate(
        self, venue
    ):
        request = AccountStateRequest(symbols=(ETHUSDC_ON_USDT,), positions=True)

        account_state = await read_account_state(venue, request, EXECUTIONS_SINCE)

        assert venue.starts() == ["positions"]
        assert account_state.conversion_rates == {}

    async def test_balances_read_the_rate_of_a_symbol_quoted_outside_its_margin(
        self, venue
    ):
        request = AccountStateRequest(symbols=(ETHUSDC_ON_USDT,), balances=True)

        account_state = await read_account_state(venue, request, EXECUTIONS_SINCE)

        assert account_state.conversion_rates == {ETHUSDC_ON_USDT: 1.5}

    async def test_a_request_for_open_orders_alone_starts_no_other_read(self, venue):
        request = AccountStateRequest(symbols=(BTCUSDT,), open_orders=True)

        await read_account_state(venue, request, EXECUTIONS_SINCE)

        assert venue.starts() == ["open_orders"]

    async def test_the_default_of_the_protocol_composes_the_same_state(self, venue):
        account_state = await venue.get_account_state(FULL_REQUEST, EXECUTIONS_SINCE)

        assert account_state.equities == {"USDT": 7.0}
        assert account_state.executions == []
