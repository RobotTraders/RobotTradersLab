from collections.abc import Callable
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab._core import AccountRequirements, read_instruments
from robottraderslab.exceptions import ExchangeRecoverableError, SymbolNotListedError
from robottraderslab.exchanges import Instrument
from robottraderslab.futures import FuturesAccount

DOGE_USDT = Symbol.create("DOGE/USDT:USDT")
BTC_USDT = Symbol.create("BTC/USDT:USDT")


def _instrument(step: float, minimum: float | None) -> Instrument:
    return Instrument(
        quantity_step=step,
        minimum_order_value=minimum,
        maximum_leverage=None,
        maximum_order_quantity=None,
        maximum_market_order_quantity=None,
    )


class TestOrderQuantity:
    def test_a_quantity_under_one_step(self):
        instrument = _instrument(step=1.0, minimum=None)

        with pytest.raises(
            ExchangeRecoverableError,
            match="0.4 of DOGE/USDT:USDT is under one quantity step of 1",
        ):
            instrument.order_quantity(DOGE_USDT, 0.4, price=None, reduce_only=True)

    def test_a_quantity_between_two_steps_is_cut_down(self):
        instrument = _instrument(step=1.0, minimum=None)

        order_quantity = instrument.order_quantity(
            DOGE_USDT, 2.7, price=None, reduce_only=False
        )

        assert order_quantity == 2.0

    def test_a_quantity_a_float_ulp_under_a_step_is_cut_to_the_step_below(self):
        instrument = _instrument(step=0.1, minimum=None)

        order_quantity = instrument.order_quantity(
            DOGE_USDT, 0.7 - 0.4, price=None, reduce_only=True
        )

        assert order_quantity == 0.2

    def test_a_quantity_the_venue_reported_keeps_its_steps(self):
        instrument = _instrument(step=0.0001, minimum=None)

        order_quantity = instrument.order_quantity(
            BTC_USDT, float("0.0003"), price=None, reduce_only=True
        )

        assert order_quantity == 0.0003

    def test_an_entry_measured_on_the_steps_the_venue_fills(self):
        instrument = _instrument(step=1.0, minimum=5.0)

        with pytest.raises(ExchangeRecoverableError, match="worth 4, under"):
            instrument.order_quantity(DOGE_USDT, 4.9, price=1.0, reduce_only=False)

    def test_an_entry_worth_exactly_the_minimum_order_value(self):
        instrument = _instrument(step=1.0, minimum=5.0)

        order_quantity = instrument.order_quantity(
            DOGE_USDT, 5.0, price=1.0, reduce_only=False
        )

        assert order_quantity == 5.0

    def test_a_reduce_only_order_worth_under_the_minimum_order_value(self):
        instrument = _instrument(step=1.0, minimum=5.0)

        order_quantity = instrument.order_quantity(
            DOGE_USDT, 4.0, price=1.0, reduce_only=True
        )

        assert order_quantity == 4.0

    def test_an_order_without_a_price_held_to_the_step_alone(self):
        instrument = _instrument(step=1.0, minimum=5.0)

        order_quantity = instrument.order_quantity(
            DOGE_USDT, 4.0, price=None, reduce_only=False
        )

        assert order_quantity == 4.0


class TestReadInstruments:
    @pytest.fixture
    def venue(self, make_venue: Callable[..., Mock]) -> Mock:
        venue = make_venue(quantity_step=1.0)
        venue.get_quote_conversion_rate.return_value = 1.0
        return venue

    @pytest.fixture
    def declared(self, venue: Mock) -> tuple[FuturesAccount, AccountRequirements]:
        account = FuturesAccount(venue, name="declared")
        requirements = AccountRequirements()
        requirements.add(account, symbols=[DOGE_USDT])
        return account, requirements

    async def test_every_declared_account_holds_the_rules_of_its_symbols(
        self, declared
    ):
        account, requirements = declared

        await read_instruments(requirements)

        assert account.long_entry(DOGE_USDT, 2.7).build().quantity == 2.0

    async def test_a_symbol_listed_on_a_later_read(self, declared, venue):
        account, requirements = declared
        listed = venue.get_instrument.return_value
        venue.get_instrument.side_effect = [SymbolNotListedError("not yet"), listed]

        await read_instruments(requirements)
        await read_instruments(requirements)

        assert account.long_entry(DOGE_USDT, 2.7).build().denial is None

    async def test_a_symbol_delisted_on_a_later_read(self, declared, venue):
        account, requirements = declared
        listed = venue.get_instrument.return_value
        venue.get_instrument.side_effect = [
            listed,
            SymbolNotListedError("Venue lists no perpetual for DOGE/USDT:USDT"),
        ]

        await read_instruments(requirements)
        await read_instruments(requirements)

        assert account.long_entry(DOGE_USDT, 2.7).build().denial == (
            "Venue lists no perpetual for DOGE/USDT:USDT"
        )
