import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from dry_run_helpers import sent
from live_bot_helpers import (
    BOOKED,
    BOT_FOLDER,
    POSITION_STRATEGY,
    SEEN_POSITIONS,
    LiveProbe,
    booked_symbols,
    bot_toml,
    run_before_slot,
    save,
    save_bot,
    section,
    seed_old_copy,
    write_draft,
)
from live_cycle_helpers import BTC, ETH, long_position
from typer.testing import CliRunner, Result

from robottraderslab._internal import hold_instance_lock, scheduler_file
from robottraderslab.bootstrap import EXCHANGE_ENTRY_POINT_GROUP
from robottraderslab.cli import app
from robottraderslab.exceptions import ExchangeRecoverableError
from robottraderslab_core.engine import bot_segment
from robottraderslab_core.exchanges import BaseExchangeAction

NOTIFIER_GROUP = "robot_traders_lab.notifiers"
LOG_HANDLER_GROUP = "robot_traders_lab.logging_handlers"
EDIT_REFUSED = "RTLAB_EDIT_REFUSED"
NOON = datetime(2026, 1, 1, 12, tzinfo=UTC)
MIDNIGHT = datetime(2026, 1, 1, tzinfo=UTC)
FAST_RETRY = "[live.retry]\nmax_attempts = 1\nbase_delay_seconds = 0\n"
NOT_CHECKED = "Not checked, run it again: "
BTC_POSITION_QUANTITY = 3.0
DRAFT_QUANTITY = 2.0


class Subscriber:
    """A notifier double recording every notification it receives."""

    def __init__(self) -> None:
        self._received: list[Any] = []

    @property
    def received(self) -> list[Any]:
        return self._received

    async def __call__(self, fill: Any) -> None:
        self._received.append(fill)

    async def notify_placement(self, placement: Any) -> None:
        self._received.append(placement)


@pytest.fixture
def dry_run(cli: CliRunner, bot_folder: Path) -> Callable[..., Result]:
    """Run `rtlab live --dry-run` on the bot's folder, on the draft when given."""

    def run(draft: Path | None = None) -> Result:
        command_args = [] if draft is None else ["--file", str(draft)]
        return cli.invoke(app, ["live", str(bot_folder), "--dry-run", *command_args])

    return run


def test_a_valid_bot_passes_and_sends_nothing(
    dry_run, bot_folder, venue, make_registered_adapter
):
    subscriber = Subscriber()
    built: list[str] = []

    def build_notifier(**settings: Any) -> Subscriber:
        built.append("notifier")
        return subscriber

    def build_log_handler(**settings: Any) -> logging.Handler:
        built.append("log handler")
        return logging.NullHandler()

    make_registered_adapter(build_notifier, "recorder", group=NOTIFIER_GROUP)
    make_registered_adapter(build_log_handler, "recorder", group=LOG_HANDLER_GROUP)
    save_bot(
        bot_folder,
        section(BTC, quantity=1.0, leverage=2.0, margin_mode="cross"),
        section(ETH),
        tail="[live.notifier.recorder.placements]\n[live.notifier.recorder.fills]\n"
        "[live.notifier.recorder.log]\n",
    )

    invoked = dry_run()

    assert invoked.exit_code == 0
    assert invoked.stdout == f"`{bot_folder / 'bot.toml'}` passes the dry run\n"
    assert invoked.stderr == ""
    assert booked_symbols() == [str(BTC), str(ETH)]
    assert sent(venue) == []
    assert built == []
    assert subscriber.received == []


@pytest.mark.parametrize(
    ("moment", "sections", "lines"),
    [
        (
            MIDNIGHT,
            (section(BTC, refused=True), section(ETH, fail="booking")),
            [
                f"{BTC}: LiveProbe: profile 1 ({BTC} 1d): this section is refused; "
                "this profile sits out the run",
                f"{ETH}: Profile {ETH}@1d sat out this candle: "
                "the booking slot raised ValueError: boom",
            ],
        ),
        (
            MIDNIGHT,
            (section(BTC, fail="signals"), section(ETH, fail="booking")),
            [
                f"{BTC}: Profile {BTC}@1d sat out this candle: "
                "the signals slot raised ValueError: boom",
                f"{ETH}: Profile {ETH}@1d sat out this candle: "
                "the booking slot raised ValueError: boom",
            ],
        ),
        (
            NOON,
            (section(BTC, fail="booking"), section(ETH, "1h", fail="signals")),
            [
                f"{ETH}: Profile {ETH}@1h sat out this candle: "
                "the signals slot raised ValueError: boom",
                f"{BTC}: Profile {BTC}@1d sat out this candle: "
                "the booking slot raised ValueError: boom",
            ],
        ),
    ],
    ids=["refused section", "signal slot", "booking slot of an open timeframe"],
)
def test_every_failure_of_the_file_is_printed_with_its_symbol_in_the_cycles_sentence(
    moment, sections, lines, dry_run, bot_folder, time_machine
):
    time_machine.move_to(moment, tick=False)
    save_bot(bot_folder, *sections)

    invoked = dry_run()

    assert invoked.exit_code == 1
    assert invoked.stdout == ""
    assert invoked.stderr.splitlines() == lines


@pytest.mark.parametrize(
    ("stopping", "options", "refused", "stop"),
    [
        (
            (section(BTC, refused=True), section(ETH, refused=True)),
            {},
            [BTC, ETH],
            "every profile",
        ),
        (
            (section(BTC, refused=True), section("NOT-A-SYMBOL", refused=True)),
            {},
            [BTC],
            "no symbol that can be read",
        ),
        (
            (section(BTC, refused=True), section(ETH, critical=True)),
            {},
            [BTC],
            "this section stops the bot",
        ),
        (
            (section(BTC, refused=True), section(ETH, venue_critical=True)),
            {},
            [BTC],
            "this section meets a venue refusal",
        ),
        (
            (section(BTC, refused=True), section(ETH)),
            {"strategy_keys": "no_such_setting = 1"},
            [BTC],
            "unknown [strategy] key no_such_setting",
        ),
    ],
    ids=[
        "every section refused",
        "refused symbol unreadable",
        "later section critical",
        "later section venue critical",
        "strategy key refused",
    ],
)
def test_the_refusals_are_printed_ahead_of_the_stop_of_the_build(
    stopping, options, refused, stop, dry_run, bot_folder
):
    save_bot(bot_folder, *stopping, **options)

    invoked = dry_run()

    assert invoked.exit_code == 1
    *refusal_lines, stop_line = invoked.stderr.splitlines()
    named = [line.split(": ")[0] for line in refusal_lines]
    assert named == [str(symbol) for symbol in refused]
    assert stop_line.startswith(f"{bot_folder.name}: ")
    assert stop in stop_line


def test_a_critical_error_of_the_venue_follows_the_failures_met_before_it(
    dry_run, bot_folder
):
    save_bot(
        bot_folder,
        section(BTC, refused=True),
        section(ETH, fail="booking:ExchangeCriticalError"),
    )

    invoked = dry_run()

    assert invoked.exit_code == 1
    first, second = invoked.stderr.splitlines()
    assert first.startswith(f"{BTC}: ")
    assert second.startswith(f"{bot_folder.name}: ExchangeCriticalError")


@pytest.mark.parametrize(
    ("bot_file_text", "cause"),
    [
        (bot_toml(section(BTC)) + "[broken\n", "TOML"),
        (bot_toml(section(BTC), strategy_keys="no_such_setting = 1"), "no_such"),
        (bot_toml(section(BTC), head='secrets_file = "gone.toml"\n'), "gone.toml"),
        (
            bot_toml(section(BTC))
            + "\n[orders.exit.params.simulator]\nreduceOnly = true\n",
            "reduceOnly",
        ),
        (bot_toml(section(BTC, fail="setup")), "boom"),
    ],
    ids=[
        "not TOML",
        "refused strategy key",
        "missing secrets file",
        "reserved order field",
        "setup raises",
    ],
)
def test_a_cause_that_stops_the_file_is_named_in_a_line(
    bot_file_text, cause, dry_run, bot_folder
):
    save(bot_folder, bot_file_text)

    invoked = dry_run()

    assert invoked.exit_code == 1
    assert invoked.stdout == ""
    assert len(invoked.stderr.splitlines()) == 1
    assert cause in invoked.stderr


@pytest.mark.parametrize(
    "bot_file_text",
    [
        bot_toml(section(BTC)) + "[broken\n",
        bot_toml(section(BTC), strategy_keys="no_such_setting = 1"),
    ],
    ids=["not TOML", "refused strategy key"],
)
def test_a_draft_that_fails_is_named_as_the_file_checked(
    bot_file_text, dry_run, bot_folder, tmp_path
):
    save_bot(bot_folder, section(BTC))
    draft = tmp_path / "bot.draft.toml"
    draft.write_text(bot_file_text)

    invoked = dry_run(draft)

    assert invoked.exit_code == 1
    assert f"`{draft}`, loaded as `bot.toml`: " in invoked.stderr


def test_a_bot_file_given_as_the_bot_is_refused_naming_its_folder(cli, bot_folder):
    save_bot(bot_folder, section(BTC))

    invoked = cli.invoke(app, ["live", str(bot_folder / "bot.toml"), "--dry-run"])

    assert invoked.exit_code == 1
    assert invoked.stderr.startswith(f"{bot_folder.name}: ")
    assert f"pass its folder, `{bot_folder}`" in invoked.stderr


def test_a_folder_with_no_bot_file_is_refused(dry_run, bot_folder):
    invoked = dry_run()

    assert invoked.exit_code == 1
    assert f"{bot_folder.name}: " in invoked.stderr
    assert f"There is no `bot.toml` in `{bot_folder}`" in invoked.stderr


def test_the_account_state_read_that_failed_is_not_checked(dry_run, bot_folder, venue):
    save_bot(bot_folder, section(BTC), tail=FAST_RETRY)
    venue.get_open_positions.side_effect = ExchangeRecoverableError(
        "the venue did not answer"
    )

    invoked = dry_run()

    assert invoked.exit_code == 1
    assert invoked.stderr.splitlines() == [
        f"{NOT_CHECKED}{bot_folder.name}: "
        "Account state fetch failed, skipping cycle: the venue did not answer"
    ]


def test_a_line_the_run_logs_never_reaches_the_terminal(
    dry_run, bot_folder, make_registered_adapter, monkeypatch
):
    def refuse(**settings: Any) -> None:
        raise ExchangeRecoverableError("the venue did not answer")

    make_registered_adapter(refuse, "unanswering", group=EXCHANGE_ENTRY_POINT_GROUP)
    save_bot(bot_folder, section(BTC), exchange="unanswering", tail=FAST_RETRY)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])

    invoked = dry_run()

    assert invoked.exit_code == 1
    assert len(invoked.stderr.splitlines()) == 1
    assert "the venue did not answer" in invoked.stderr


def test_a_candle_that_could_not_be_read_is_not_checked(dry_run, bot_folder):
    save_bot(bot_folder, section(BTC, fail="signals:MissingOhlcvDataError"))

    invoked = dry_run()

    assert invoked.exit_code == 1
    assert invoked.stderr.startswith(f"{NOT_CHECKED}{BTC}: ")


def test_a_failure_and_a_read_that_failed_are_both_printed(dry_run, bot_folder):
    save_bot(
        bot_folder,
        section(BTC, fail="signals:MissingOhlcvDataError"),
        section(ETH, fail="booking"),
    )

    invoked = dry_run()

    assert invoked.exit_code == 1
    lines = invoked.stderr.splitlines()
    assert sorted(line.startswith(NOT_CHECKED) for line in lines) == [False, True]


def test_a_draft_is_read_as_the_bot_holding_a_position(
    dry_run, bot_folder, venue, tmp_path
):
    original = save_bot(
        bot_folder, section(BTC, quantity=1.0), strategy=POSITION_STRATEGY
    )
    draft = write_draft(
        tmp_path / "elsewhere" / "draft.toml",
        section(BTC, quantity=DRAFT_QUANTITY),
        strategy=POSITION_STRATEGY,
    )
    venue.get_open_positions.return_value = {
        BTC: long_position(BTC, BTC_POSITION_QUANTITY)
    }

    invoked = dry_run(draft)

    assert invoked.exit_code == 0
    assert invoked.stdout == f"`{draft}` passes the dry run\n"
    assert [position.quantity for position in SEEN_POSITIONS] == [BTC_POSITION_QUANTITY]
    assert [booked.quantity for booked in BOOKED] == [DRAFT_QUANTITY]
    assert (bot_folder / "bot.toml").read_bytes() == original.encode()


def test_a_relative_draft_starts_from_the_bots_folder(
    cli, bot_folder, venue, monkeypatch, tmp_path
):
    save_bot(bot_folder, section(BTC, quantity=1.0), strategy=POSITION_STRATEGY)
    write_draft(
        bot_folder / "bot.draft.toml",
        section(BTC, quantity=DRAFT_QUANTITY),
        strategy=POSITION_STRATEGY,
    )
    venue.get_open_positions.return_value = {
        BTC: long_position(BTC, BTC_POSITION_QUANTITY)
    }
    monkeypatch.chdir(tmp_path)

    invoked = cli.invoke(
        app, ["live", BOT_FOLDER, "--dry-run", "--file", "bot.draft.toml"]
    )

    assert invoked.exit_code == 0
    expected_path = Path(BOT_FOLDER, "bot.draft.toml")
    assert invoked.stdout == f"`{expected_path}` passes the dry run\n"
    assert [position.quantity for position in SEEN_POSITIONS] == [BTC_POSITION_QUANTITY]
    assert [booked.quantity for booked in BOOKED] == [DRAFT_QUANTITY]


def test_a_draft_that_does_not_exist_is_a_failure_naming_it(
    dry_run, bot_folder, tmp_path
):
    save_bot(bot_folder, section(BTC))
    missing = tmp_path / "gone.draft.toml"

    invoked = dry_run(missing)

    assert invoked.exit_code == 1
    assert invoked.stderr.startswith(f"{bot_folder.name}: FileNotFoundError")
    assert missing.name in invoked.stderr


def test_the_order_ids_of_a_draft_carry_the_bots_segment(dry_run, bot_folder, tmp_path):
    save_bot(bot_folder, section(BTC))
    draft = write_draft(tmp_path / "other-name" / "draft.toml", section(BTC))

    dry_run(draft)

    assert (
        bot_segment(bot_folder.name) in BaseExchangeAction(symbol=BTC).client_order_id
    )


def test_a_file_without_the_dry_run_is_a_usage_error_that_touches_no_venue(
    cli, bot_folder, venue, tmp_path
):
    save_bot(bot_folder, section(BTC))
    draft = write_draft(tmp_path / "draft.toml", section(BTC))

    invoked = cli.invoke(app, ["live", str(bot_folder), "--file", str(draft)])

    assert invoked.exit_code == 2
    assert "--file goes with --dry-run" in invoked.stderr
    venue.get_open_positions.assert_not_called()
    assert sent(venue) == []
    assert not (bot_folder / ".rtlab").exists()


def test_an_existing_working_copy_and_every_file_stay_as_they_were(
    dry_run, bot_folder, tmp_path
):
    save_bot(bot_folder, section(BTC))
    seed_old_copy(bot_folder)
    save_bot(bot_folder, section(BTC, quantity=2.0))
    launch_memory = scheduler_file(tmp_path / "registry.toml", ".launches.json")
    launch_memory.parent.mkdir()
    launch_memory.write_text("{}")

    with _paths_unchanged(tmp_path):
        invoked = dry_run()

    assert invoked.exit_code == 0


def test_no_working_copy_log_or_lock_is_created_where_none_stood(
    dry_run, bot_folder, tmp_path
):
    save_bot(bot_folder, section(BTC))

    with _paths_unchanged(tmp_path):
        invoked = dry_run()

    assert invoked.exit_code == 0


def test_the_command_runs_while_another_process_holds_the_instance_lock(
    dry_run, bot_folder
):
    save_bot(bot_folder, section(BTC))

    with hold_instance_lock(bot_folder / "bot.toml"):
        invoked = dry_run()

    assert invoked.exit_code == 0
    assert "passes the dry run" in invoked.stdout


def test_a_cycle_starting_during_the_dry_run_is_not_refused(
    dry_run, bot_folder, monkeypatch
):
    save_bot(bot_folder, section(BTC))
    held_by_a_cycle: list[bool] = []

    def start_a_cycle() -> None:
        with hold_instance_lock(bot_folder / "bot.toml") as held:
            held_by_a_cycle.append(held)

    run_before_slot(monkeypatch, LiveProbe, "generate_general_signals", start_a_cycle)

    invoked = dry_run()

    assert invoked.exit_code == 0
    assert held_by_a_cycle == [True]


def test_a_refused_edit_in_the_environment_changes_nothing(
    dry_run, bot_folder, monkeypatch
):
    save_bot(bot_folder, section(BTC))
    monkeypatch.setenv(EDIT_REFUSED, "an edit refused by the scheduler")

    invoked = dry_run()

    assert invoked.exit_code == 0
    assert invoked.stdout == f"`{bot_folder / 'bot.toml'}` passes the dry run\n"
    assert invoked.stderr == ""


@contextmanager
def _paths_unchanged(root: Path) -> Iterator[None]:
    """Assert that no file or folder under the root changed while the block ran."""
    before = _paths_under(root)
    yield
    assert _paths_under(root) == before


def _paths_under(root: Path) -> dict[str, tuple[bytes | None, int]]:
    """Every file and folder under the root, with its bytes and modification time.

    The bytes are None for a folder, and the modification time is in nanoseconds.
    """
    return {
        str(path.relative_to(root)): (
            path.read_bytes() if path.is_file() else None,
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
    }
