import logging
import string
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from dry_run_helpers import MESSAGE
from live_bot_helpers import (
    BOOKED,
    CLOSING_BTC,
    DOUBLED_BTC,
    FAILING_VENUE,
    PLAIN_STRATEGY,
    REFUSED_BTC,
    SOL,
    THREE,
    TWO,
    LiveProbe,
    booked_symbols,
    bot_toml,
    closes,
    copies_at_each_call,
    copies_at_each_order,
    copy_of,
    cycle,
    entries,
    entry_symbols,
    failing_the_first_call,
    logged,
    plain_signal_runs,
    read_copy,
    refuse_copy_writes,
    run_before_slot,
    save,
    save_bot,
    saving,
    section,
    seed_bot,
    seeded,
    signalled_symbols,
)
from live_cycle_helpers import BTC, ETH, long_position

from robottraderslab._internal import EX_TEMPFAIL, hold_instance_lock
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import BotCriticalError
from robottraderslab.live import run_livebot
from robottraderslab_core.engine import bot_segment, name_process
from robottraderslab_core.exchanges import BaseExchangeAction

NOTIFIER_GROUP = "robot_traders_lab.notifiers"
EDIT_REFUSED = "RTLAB_EDIT_REFUSED"
REFUSAL = (
    "`my/bot.toml` adds SOL/USDT:USDT on the account `my_account`, "
    "which `other` already trades; "
    "remove it from one of the two bots or give each its own account"
)
MOMENT = datetime(2026, 1, 1, tzinfo=UTC)
A_MINUTE_LATER = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
NO_LIVE_SECTION = (
    'secrets_file = "creds.toml"\n\n[strategy]\nstrategy_class = "any.Strategy"\n'
)


@pytest.fixture
def slow_dry_run(monkeypatch: pytest.MonkeyPatch, time_machine: Any) -> None:
    """The dry run takes a minute: the clock moves on as its first general slot runs."""
    run_before_slot(
        monkeypatch,
        LiveProbe,
        "generate_general_signals",
        partial(time_machine.move_to, A_MINUTE_LATER, tick=False),
    )


def test_a_first_run_copies_a_valid_file_before_its_first_order(bot_folder, venue):
    text = save_bot(bot_folder, *TWO)
    seen_at_order = copies_at_each_order(bot_folder, venue)

    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert seen_at_order[0] == text
    assert logged(bot_folder, "WARNING") == []
    assert logged(bot_folder, "ERROR") == []


def test_a_first_run_whose_file_does_not_pass_writes_no_copy(bot_folder, venue):
    save_bot(bot_folder, *REFUSED_BTC)

    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert not copy_of(bot_folder).exists()
    assert entry_symbols(venue) == {str(ETH)}


def test_an_unchanged_file_runs_each_slot_once_and_writes_nothing(bot_folder, venue):
    seed_bot(bot_folder, section(BTC, fail="booking"), section(ETH))
    written_at = copy_of(bot_folder).stat().st_mtime_ns

    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert signalled_symbols() == [str(BTC), str(ETH)]
    assert booked_symbols() == [str(BTC), str(ETH)]
    assert entry_symbols(venue) == {str(ETH)}
    assert copy_of(bot_folder).stat().st_mtime_ns == written_at


def test_a_dry_run_sends_nothing_before_the_copy_is_written(
    bot_folder, venue, make_registered_adapter
):
    seed_bot(bot_folder, *TWO)
    edited = save_bot(
        bot_folder,
        section(BTC, quantity=2.0, leverage=2.0, margin_mode="cross"),
        section(ETH),
        tail="[live.notifier.recorder.placements]\n",
    )
    notifier = CopyRecorder(bot_folder)
    make_registered_adapter(
        lambda **settings: notifier, "recorder", group=NOTIFIER_GROUP
    )
    copied_at_leverage = copies_at_each_call(bot_folder, venue.set_leverage)
    copied_at_margin_mode = copies_at_each_call(bot_folder, venue.set_margin_mode)
    copied_at_order = copies_at_each_order(bot_folder, venue)

    cycle(bot_folder)

    assert copied_at_leverage == [edited]
    assert copied_at_margin_mode == [edited]
    assert copied_at_order == [edited, edited]
    assert notifier.copies == [edited, edited]


def test_a_profile_whose_candle_does_not_close_is_booked_once_in_the_dry_run(
    bot_folder, venue, time_machine
):
    time_machine.move_to(datetime(2026, 1, 1, 1, 0, tzinfo=UTC), tick=False)
    seed_bot(bot_folder, section(BTC, "4h"), section(ETH, "4h"))
    edited = save_bot(bot_folder, section(BTC, "4h", quantity=2.0), section(ETH, "4h"))

    cycle(bot_folder)

    assert booked_symbols() == [str(BTC), str(ETH)]
    assert entries(venue) == []
    assert read_copy(bot_folder) == edited


def test_a_dry_run_sizes_a_new_leverage_as_the_cycle_then_sends_it(bot_folder, venue):
    seed_bot(bot_folder, *TWO)
    save_bot(bot_folder, section(BTC, leverage=3.0), section(ETH))

    cycle(bot_folder)

    assert BOOKED[0].symbol == BTC
    assert BOOKED[0].quantity == 3.0
    assert (str(BTC), 3.0) in entries(venue)


@pytest.mark.parametrize(
    "btc_fields",
    [
        {"unknown_key": 1},
        {"quantity": "many"},
        {"refused": True},
        {"raises": True},
        {"fail": "signals"},
        {"fail": "booking"},
    ],
    ids=[
        "unknown key",
        "wrong type",
        "ProfileConfigError",
        "raised in __post_init__",
        "signal slot raises",
        "booking slot raises",
    ],
)
def test_a_section_that_fails_runs_its_symbol_from_the_copy_in_exit_only_mode(
    bot_folder, venue, btc_fields
):
    working_copy = seed_bot(bot_folder, *CLOSING_BTC)
    venue.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
    save_bot(bot_folder, section(BTC, closes=True, **btc_fields), section(ETH))

    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert closes(venue) == [str(BTC)]
    assert entry_symbols(venue) == {str(ETH)}
    assert read_copy(bot_folder) == working_copy
    assert any(str(BTC) in line for line in logged(bot_folder, "ERROR")[:-1])
    assert f"with {BTC} in exit-only mode" in logged(bot_folder, "ERROR")[-1]


def test_two_failures_on_two_symbols_are_both_logged(bot_folder, venue):
    seed_bot(bot_folder, *THREE)
    save_bot(
        bot_folder,
        section(BTC, refused=True),
        section(ETH, fail="booking"),
        section(SOL),
    )

    cycle(bot_folder)

    errors = logged(bot_folder, "ERROR")
    assert len(errors) == 3
    assert f"with {BTC}, {ETH} in exit-only mode" in errors[-1]
    assert entry_symbols(venue) == {str(SOL)}


@pytest.mark.parametrize(
    ("eth_section", "stop_text"),
    [
        (section(ETH, fail="booking:ExchangeCriticalError"), "ExchangeCriticalError"),
        (section(ETH, critical=True), "this section stops the bot"),
    ],
    ids=["critical error of the venue", "critical error of a section"],
)
def test_the_refusals_met_before_a_critical_error_are_logged_ahead_of_it(
    eth_section, stop_text, bot_folder
):
    working_copy = seed_bot(bot_folder, *TWO)
    save_bot(bot_folder, section(BTC, refused=True), eth_section)

    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert read_copy(bot_folder) == working_copy
    section_refusal, stop_line, exit_only = logged(bot_folder, "ERROR")
    assert f"{BTC}: " in section_refusal
    assert stop_text in stop_line
    assert f"with {BTC}, {ETH} in exit-only mode" in exit_only


def test_a_failure_beside_a_read_that_failed_is_logged_at_error(bot_folder, venue):
    seed_bot(bot_folder, *TWO)
    save_bot(
        bot_folder,
        section(BTC, fail="signals:MissingOhlcvDataError"),
        section(ETH, fail="booking"),
    )

    cycle(bot_folder)

    errors = logged(bot_folder, "ERROR")
    assert len(errors) == 3
    assert f"with {BTC}, {ETH} in exit-only mode" in errors[-1]
    assert logged(bot_folder, "WARNING") == []


@pytest.mark.parametrize(
    ("edit", "cause"),
    [
        (saving(bot_toml(*TWO) + "[broken\n"), "TOML"),
        (saving(bot_toml(*TWO).encode("utf-16")), "UTF-8"),
        (lambda folder: (folder / "bot.toml").unlink(), "There is no `bot.toml`"),
        (saving(bot_toml(*TWO).replace("symbol =", "symbl =", 1)), "symbol"),
        (saving(bot_toml(*TWO).replace(str(BTC), "BTC")), "BTC"),
        (saving(bot_toml(*TWO, strategy_keys="no_such_setting = 1")), "no_such"),
        (saving(bot_toml(*TWO, tail="no_such_live_key = 1\n")), "no_such_live"),
        (saving(bot_toml(*TWO, head='secrets_file = "gone.toml"\n')), "my_account"),
        (
            saving(
                bot_toml(*TWO, tail='[live.logging]\nlogfile = "creds.toml/x.log"\n')
            ),
            "creds.toml",
        ),
        (saving(bot_toml(*TWO, exchange=FAILING_VENUE)), "refuses these keys"),
        (
            saving(
                bot_toml(*TWO) + "\n[orders.exit.params.simulator]\nreduceOnly = true\n"
            ),
            "reduceOnly",
        ),
    ],
    ids=[
        "not TOML",
        "UTF-16",
        "no bot.toml",
        "symbl",
        "unreadable symbol",
        "strategy key",
        "live key",
        "missing secrets file",
        "log folder",
        "account load",
        "reserved venue field",
    ],
)
def test_a_file_wide_failure_runs_every_symbol_from_the_copy_in_exit_only_mode(
    bot_folder, venue, edit, cause
):
    working_copy = seed_bot(bot_folder, *TWO)
    edit(bot_folder)

    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert entries(venue) == []
    assert read_copy(bot_folder) == working_copy
    errors = logged(bot_folder, "ERROR")
    assert len(errors) == 2
    assert cause in errors[0]
    assert "with every symbol in exit-only mode" in errors[1]


@pytest.mark.parametrize(
    ("btc_sections", "cause"),
    [
        ((section(BTC, fail="setup"),), MESSAGE),
        ((section(BTC, fail="general_signals"),), MESSAGE),
        ((section(BTC, fail="final_booking"),), MESSAGE),
        ((section(BTC, fail="booking:BotCriticalError"),), MESSAGE),
        ((section(BTC), section(BTC)), "profile_id"),
    ],
    ids=[
        "setup raises",
        "general slot",
        "final slot",
        "booking critical",
        "section added under another's id",
    ],
)
def test_a_failure_of_one_symbols_edit_runs_it_alone_in_exit_only_mode(
    bot_folder, venue, btc_sections, cause
):
    working_copy = seed_bot(bot_folder, *TWO)
    save_bot(bot_folder, *btc_sections, section(ETH))

    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert entry_symbols(venue) == {str(ETH)}
    assert read_copy(bot_folder) == working_copy
    errors = logged(bot_folder, "ERROR")
    assert cause in errors[0]
    assert f"with {BTC} in exit-only mode" in errors[-1]


def test_a_symbol_the_edit_removes_beside_a_broken_one_runs_in_exit_only_mode(
    bot_folder, venue
):
    seed_bot(bot_folder, section(BTC), section(ETH, closes=True))
    venue.get_open_positions.return_value = {ETH: long_position(ETH, 1.0)}
    save_bot(bot_folder, section(BTC, refused=True))

    cycle(bot_folder)

    assert closes(venue) == [str(ETH)]
    assert entries(venue) == []
    assert f"with {BTC}, {ETH} in exit-only mode" in logged(bot_folder, "ERROR")[-1]


def test_a_valid_change_waits_beside_a_broken_one_then_trades_once_corrected(
    bot_folder, venue
):
    seed_bot(bot_folder, *THREE)
    save_bot(
        bot_folder,
        section(BTC, quantity=5.0),
        section(ETH, refused=True),
        section(SOL),
    )

    cycle(bot_folder)
    waiting = entries(venue)
    venue.place_market_order.reset_mock()
    errors_while_broken = len(logged(bot_folder, "ERROR"))
    corrected = save_bot(
        bot_folder, section(BTC, quantity=5.0), section(ETH), section(SOL)
    )
    cycle(bot_folder)

    assert waiting == [(str(SOL), 1.0)]
    assert (str(BTC), 5.0) in entries(venue)
    assert read_copy(bot_folder) == corrected
    assert len(logged(bot_folder, "ERROR")) == errors_while_broken


def test_a_symbol_added_beside_a_refused_section_waits_for_the_correction(
    bot_folder, venue
):
    seed_bot(bot_folder, *TWO)
    save_bot(bot_folder, section(BTC), section(ETH, refused=True), section(SOL))

    cycle(bot_folder)
    signalled_while_broken = signalled_symbols()
    sent_while_broken = entry_symbols(venue)
    save_bot(bot_folder, *THREE)
    cycle(bot_folder)

    assert signalled_while_broken.count(str(SOL)) == 1
    assert str(SOL) not in sent_while_broken
    assert str(SOL) in entry_symbols(venue)


def test_a_slot_failing_in_both_files_on_an_untouched_symbol_lets_the_file_pass(
    bot_folder,
):
    seed_bot(bot_folder, section(BTC), section(ETH, fail="booking"))
    edited = save_bot(
        bot_folder, section(BTC, quantity=2.0), section(ETH, fail="booking")
    )

    cycle(bot_folder)

    assert read_copy(bot_folder) == edited


@pytest.mark.parametrize("error", ["MissingOhlcvDataError", "ExchangeRecoverableError"])
def test_a_read_that_failed_leaves_the_file_unverified_at_warning(bot_folder, error):
    working_copy = seed_bot(bot_folder, *TWO)
    save_bot(bot_folder, section(BTC, fail=f"signals:{error}"), section(ETH))

    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert read_copy(bot_folder) == working_copy
    assert logged(bot_folder, "ERROR") == []
    assert f"with {BTC} in exit-only mode" in logged(bot_folder, "WARNING")[-1]


def test_a_file_replaced_during_the_dry_run_is_not_what_the_cycle_copies(
    bot_folder, venue, monkeypatch
):
    seed_bot(bot_folder, *TWO)
    edited = save_bot(bot_folder, *DOUBLED_BTC)
    replaced = bot_toml(section(BTC, quantity=9.0), section(ETH))
    run_before_slot(
        monkeypatch,
        LiveProbe,
        "generate_general_signals",
        partial(save, bot_folder, replaced),
    )

    cycle(bot_folder)

    assert read_copy(bot_folder) == edited
    assert (str(BTC), 2.0) in entries(venue)


@pytest.mark.usefixtures("slow_dry_run")
def test_the_run_after_a_slow_dry_run_books_the_candles_of_the_cycles_start(
    bot_folder, venue
):
    seed_bot(bot_folder, *TWO)
    save_bot(bot_folder, *DOUBLED_BTC)

    cycle(bot_folder)

    assert entry_symbols(venue) == {str(BTC), str(ETH)}


@pytest.mark.usefixtures("slow_dry_run")
def test_the_run_after_a_slow_dry_run_names_its_orders_from_the_cycles_start(
    bot_folder, venue
):
    seed_bot(bot_folder, *TWO)
    save_bot(bot_folder, *DOUBLED_BTC)

    cycle(bot_folder)

    placed = venue.place_market_order.await_args_list[0]
    name_process(bot_folder.name, MOMENT)
    launch = BaseExchangeAction(symbol=BTC).client_order_id.rstrip(string.digits)
    assert placed.kwargs["client_order_id"].startswith(launch)


def test_the_copys_run_keeps_the_bots_secrets_log_and_order_ids(bot_folder, venue):
    seed_bot(bot_folder, *CLOSING_BTC)
    venue.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
    save(bot_folder, "[broken\n")

    cycle(bot_folder)

    assert venue.received_settings["api_key"] == "shh"
    assert (bot_folder / "logs" / "live.log").is_file()
    close = venue.place_market_order.await_args_list[0]
    assert bot_segment(bot_folder.name) in close.kwargs["client_order_id"]


def test_a_dry_run_failing_on_one_candle_passes_on_the_next(bot_folder, monkeypatch):
    seed_bot(bot_folder, *TWO)
    edited = save_bot(bot_folder, *DOUBLED_BTC)
    run_before_slot(
        monkeypatch,
        LiveProbe,
        "book_profile_actions",
        failing_the_first_call(ValueError(MESSAGE)),
    )

    cycle(bot_folder)
    copied_after_first = read_copy(bot_folder)
    cycle(bot_folder)

    assert copied_after_first != edited
    assert read_copy(bot_folder) == edited


def test_the_dry_run_logs_nothing_above_debug(bot_folder, monkeypatch):
    seed_bot(bot_folder, *TWO)
    save_bot(bot_folder, *DOUBLED_BTC)
    run_before_slot(
        monkeypatch,
        LiveProbe,
        "generate_general_signals",
        partial(logging.getLogger("probe").warning, "a slot spoke"),
    )

    cycle(bot_folder)

    assert len(logged(bot_folder, "WARNING")) == 1
    assert len(_lines_about(bot_folder, "LiveProbe: 2 profiles")) == 1


def test_a_profile_of_the_copy_whose_slot_raises_is_skipped_alone(bot_folder, venue):
    seed_bot(bot_folder, section(BTC, fail="booking"), section(ETH), section(SOL))
    save_bot(
        bot_folder,
        section(BTC, fail="booking"),
        section(ETH, refused=True),
        section(SOL),
    )

    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert entry_symbols(venue) == {str(SOL)}


def test_a_copy_the_strategy_refuses_beside_a_broken_file_stops_the_bot(bot_folder):
    seed_bot(bot_folder, *TWO, strategy_keys="no_such_setting = 1")
    save(bot_folder, "[broken\n")

    with pytest.raises(BotCriticalError, match="bot.working.toml"):
        cycle(bot_folder)


def test_a_copy_declaring_no_live_section_beside_a_broken_file_stops_the_bot(
    bot_folder,
):
    seeded(bot_folder, NO_LIVE_SECTION)
    save(bot_folder, "[broken\n")

    with pytest.raises(BotCriticalError, match="bot.working.toml"):
        cycle(bot_folder)


def test_an_unchanged_file_that_cannot_load_stops_the_bot(bot_folder, venue):
    seeded(bot_folder, NO_LIVE_SECTION)

    with pytest.raises(BotCriticalError, match=r"`\[live\]` section is required"):
        cycle(bot_folder)

    venue.place_market_order.assert_not_called()


def test_with_no_copy_a_file_that_is_not_toml_stops_the_bot(bot_folder):
    save(bot_folder, "[broken\n")

    with pytest.raises(BotCriticalError):
        cycle(bot_folder)


def test_a_copy_that_is_not_toml_counts_as_none_and_is_replaced(bot_folder):
    text = seed_bot(bot_folder, *TWO)
    copy_of(bot_folder).write_text("[broken\n")

    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert len(logged(bot_folder, "WARNING")) == 1
    assert read_copy(bot_folder) == text


def test_a_plain_strategy_is_verified_by_its_two_run_methods(bot_folder):
    seed_bot(bot_folder, strategy=PLAIN_STRATEGY)
    edited = save_bot(bot_folder, strategy=PLAIN_STRATEGY, strategy_keys='fail = ""')

    cycle(bot_folder)

    assert read_copy(bot_folder) == edited
    assert plain_signal_runs() == 2


def test_a_plain_strategy_that_fails_runs_every_symbol_in_exit_only_mode(
    bot_folder, venue
):
    working_copy = seed_bot(bot_folder, strategy=PLAIN_STRATEGY)
    save_bot(bot_folder, strategy=PLAIN_STRATEGY, strategy_keys='fail = "booking"')

    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert entries(venue) == []
    assert read_copy(bot_folder) == working_copy
    assert "with every symbol in exit-only mode" in logged(bot_folder, "ERROR")[-1]


def test_a_bot_run_from_code_keeps_no_working_copy(bot_folder, venue):
    save_bot(bot_folder, *TWO)

    run_livebot(BotConfig.from_folder(bot_folder))

    assert not copy_of(bot_folder).exists()
    assert entry_symbols(venue) == {str(BTC), str(ETH)}


def test_a_copy_that_cannot_be_written_runs_the_previous_copy(
    bot_folder, venue, monkeypatch
):
    working_copy = seed_bot(bot_folder, *TWO)
    save_bot(bot_folder, *DOUBLED_BTC)
    refuse_copy_writes(monkeypatch)

    exit_code = cycle(bot_folder)

    assert exit_code == 0
    assert read_copy(bot_folder) == working_copy
    assert "bot.working.toml" in logged(bot_folder, "ERROR")[0]
    assert f"with {BTC} in exit-only mode" in logged(bot_folder, "ERROR")[-1]
    assert logged(bot_folder, "WARNING") == []
    assert entry_symbols(venue) == {str(ETH)}


def test_an_edit_that_parses_equal_runs_every_symbol_when_its_copy_cannot_be_written(
    bot_folder, venue, monkeypatch
):
    working_copy = seed_bot(bot_folder, *TWO)
    save(bot_folder, working_copy + "# a comment\n")
    refuse_copy_writes(monkeypatch)

    cycle(bot_folder)

    assert read_copy(bot_folder) == working_copy
    assert "with no symbol in exit-only mode" in logged(bot_folder, "ERROR")[-1]
    assert entry_symbols(venue) == {str(BTC), str(ETH)}


def test_a_config_already_trading_leaves_the_copy_to_the_process_trading(
    bot_folder, venue
):
    working_copy = seed_bot(bot_folder, *TWO)
    save_bot(bot_folder, *DOUBLED_BTC)

    with hold_instance_lock(bot_folder / "bot.toml"):
        exit_code = cycle(bot_folder)

    assert exit_code == EX_TEMPFAIL
    assert read_copy(bot_folder) == working_copy
    venue.place_market_order.assert_not_called()


def test_a_refused_edit_runs_the_copy_with_no_dry_run_and_writes_no_copy(
    bot_folder, venue, monkeypatch
):
    working_copy = seed_bot(bot_folder, *TWO)
    save_bot(bot_folder, *THREE)

    exit_code = _cycle_under_refusal(bot_folder, monkeypatch)

    assert exit_code == 0
    assert read_copy(bot_folder) == working_copy
    assert signalled_symbols() == [str(BTC), str(ETH)]
    assert entry_symbols(venue) == {str(BTC), str(ETH)}
    errors = logged(bot_folder, "ERROR")
    assert len(errors) == 2
    assert f"bot.toml` is not applied: {REFUSAL}" in errors[0]
    assert f"with {SOL} in exit-only mode" in errors[1]


def test_a_refusal_beside_a_file_that_cannot_load_is_reported_before_the_load_failure(
    bot_folder, venue, monkeypatch
):
    working_copy = seed_bot(bot_folder, *TWO)
    save(bot_folder, "[broken\n")

    exit_code = _cycle_under_refusal(bot_folder, monkeypatch)

    assert exit_code == 0
    assert entries(venue) == []
    assert read_copy(bot_folder) == working_copy
    errors = logged(bot_folder, "ERROR")
    assert len(errors) == 3
    assert f"is not applied: {REFUSAL}" in errors[0]
    assert "TOML" in errors[1]
    assert "with every symbol in exit-only mode" in errors[2]


def test_a_refusal_changes_nothing_for_a_file_equal_to_the_copy(
    bot_folder, venue, monkeypatch
):
    seed_bot(bot_folder, *TWO)

    _assert_a_refusal_changes_nothing(bot_folder, venue, monkeypatch)


def test_a_refusal_changes_nothing_for_a_bot_with_no_copy(
    bot_folder, venue, monkeypatch
):
    text = save_bot(bot_folder, *TWO)

    _assert_a_refusal_changes_nothing(bot_folder, venue, monkeypatch)

    assert read_copy(bot_folder) == text


def test_an_empty_refusal_is_no_refusal(bot_folder, monkeypatch):
    seed_bot(bot_folder, *TWO)
    edited = save_bot(bot_folder, *DOUBLED_BTC)
    monkeypatch.setenv(EDIT_REFUSED, "")

    cycle(bot_folder)

    assert read_copy(bot_folder) == edited
    assert logged(bot_folder, "ERROR") == []


class CopyRecorder:
    """A notifier of placements that reads the working copy as each arrives."""

    def __init__(self, folder: Path) -> None:
        self._folder = folder
        self.copies: list[str] = []

    async def notify_placement(self, placement: Any) -> None:
        self.copies.append(read_copy(self._folder))


def _lines_about(folder: Path, text: str) -> list[str]:
    return [line for line in logged(folder, "INFO") if text in line]


def _cycle_under_refusal(bot_folder: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setenv(EDIT_REFUSED, REFUSAL)
    return cycle(bot_folder)


def _assert_a_refusal_changes_nothing(
    bot_folder: Path, venue: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code = _cycle_under_refusal(bot_folder, monkeypatch)

    assert exit_code == 0
    assert entry_symbols(venue) == {str(BTC), str(ETH)}
    assert logged(bot_folder, "ERROR") == []
