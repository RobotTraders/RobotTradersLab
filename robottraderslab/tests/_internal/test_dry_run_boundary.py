import logging
from dataclasses import dataclass

import pytest
from readouts import logged

from robottraderslab import Symbol
from robottraderslab._internal import DryRunFailure, failure_of
from robottraderslab._internal.profile_failure_boundary import (
    ProfileSlot,
    dry_run_boundary,
)
from robottraderslab.exceptions import (
    BotCriticalError,
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ExchangeTransientError,
    MissingOhlcvDataError,
    ProfileError,
)

BTC = Symbol.create("BTC/USDT:USDT")


@dataclass
class _Profile:
    profile_id: str
    timeframe: str
    symbol: Symbol | str = BTC


def test_a_profile_whose_symbol_is_a_string_has_its_failure_attributed_to_the_symbol(
    make_raiser,
):
    failures: list[DryRunFailure] = []
    profile = _Profile("p1", "1d", "BTC/USDT:USDT")

    dry_run_boundary(
        failures, ProfileSlot.BOOKING, profile, make_raiser(ValueError("boom"))
    )

    (failure,) = failures
    assert isinstance(failure.symbol, Symbol)
    assert failure.symbol == BTC


def test_a_profile_with_no_symbol_at_all_has_its_failure_attributed_to_none(
    make_raiser,
):
    class Bare:
        profile_id = "p1"
        timeframe = "1d"

    failures: list[DryRunFailure] = []

    dry_run_boundary(
        failures, ProfileSlot.SIGNALS, Bare(), make_raiser(ValueError("boom"))
    )

    (failure,) = failures
    assert failure.symbol is None


def test_a_critical_error_of_the_venue_is_not_contained(make_raiser):
    failures: list[DryRunFailure] = []

    with pytest.raises(ExchangeCriticalError):
        dry_run_boundary(
            failures,
            ProfileSlot.BOOKING,
            _Profile("p1", "1d"),
            make_raiser(ExchangeCriticalError("down")),
        )

    assert failures == []


def test_a_critical_error_of_the_bot_is_contained_as_a_failure_of_the_symbol(
    make_raiser,
):
    failures: list[DryRunFailure] = []

    dry_run_boundary(
        failures,
        ProfileSlot.BOOKING,
        _Profile("p1", "1d"),
        make_raiser(BotCriticalError("broken")),
    )

    (failure,) = failures
    assert failure.symbol == BTC


@pytest.mark.parametrize(
    ("error", "recoverable"),
    [
        (ExchangeRecoverableError("no answer"), True),
        (ExchangeTransientError("timeout"), True),
        (MissingOhlcvDataError("BTC/USDT:USDT", "1h"), True),
        (ValueError("boom"), False),
        (ProfileError("stated on purpose"), False),
        (BotCriticalError("broken"), False),
    ],
    ids=["recoverable", "transient", "missing candles", "value", "profile", "bot"],
)
def test_a_failure_is_recoverable_when_it_is_a_read_that_got_no_answer(
    error, recoverable
):
    assert failure_of(error, None).recoverable is recoverable


def test_a_failure_says_the_class_and_the_text_of_its_error():
    assert failure_of(ValueError("boom"), None).message == "ValueError: boom"


def test_the_failure_names_the_profile_the_slot_and_the_cause_as_the_live_record_does(
    make_raiser,
):
    failures: list[DryRunFailure] = []

    dry_run_boundary(
        failures,
        ProfileSlot.SIGNALS,
        _Profile("p1", "1d"),
        make_raiser(ValueError("boom")),
    )

    assert [failure.message for failure in failures] == [
        "Profile p1 sat out this candle: the signals slot raised ValueError: boom"
    ]


def test_the_failure_is_logged_at_debug_and_no_higher(caplog, make_raiser):
    failures: list[DryRunFailure] = []

    with caplog.at_level(logging.DEBUG):
        dry_run_boundary(
            failures,
            ProfileSlot.BOOKING,
            _Profile("p1", "1d"),
            make_raiser(ValueError("boom")),
        )

    assert logged(caplog) == [(logging.DEBUG, failures[0].message)]


def test_a_slot_that_raises_is_reported_as_not_run_through(make_raiser):
    ran = dry_run_boundary(
        [], ProfileSlot.BOOKING, _Profile("p1", "1d"), make_raiser(ValueError("boom"))
    )

    assert not ran


def test_a_slot_that_runs_through_reports_nothing():
    failures: list[DryRunFailure] = []

    ran = dry_run_boundary(
        failures, ProfileSlot.BOOKING, _Profile("p1", "1d"), lambda: None
    )

    assert ran
    assert failures == []
