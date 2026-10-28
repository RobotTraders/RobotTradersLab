import logging
import subprocess
import sys
from collections.abc import Callable
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from robottraderslab._core import (
    EX_CONFIG,
    EX_DATAERR,
    EX_TEMPFAIL,
    hold_instance_lock,
)
from robottraderslab.cli import app, main
from robottraderslab.exceptions import (
    BotCriticalError,
    ExchangeCriticalError,
    ExchangeTransientError,
)
from robottraderslab.scaffold import ExampleError

COMMAND_SUBPACKAGES = {
    "backtest": "robottraderslab.backtester",
    "check": "robottraderslab.live.check",
    "flatten": "robottraderslab.live.flatten",
    "grid-search": "robottraderslab.grid_search",
    "init": "robottraderslab.scaffold.workspace",
    "live": "robottraderslab.live",
    "report": "robottraderslab.live.report",
    "new": "robottraderslab.scaffold",
    "scheduler": "robottraderslab.scheduler",
}

DISPATCHES = [
    (
        "backtest",
        [
            "backtest",
            "my-bot",
            "--log-file",
            "run.log",
            "--console",
            "--no-chart",
            "--no-save",
        ],
        {
            "folder": Path("my-bot"),
            "logfile_override": Path("run.log"),
            "console": True,
            "chart": False,
            "save": False,
        },
    ),
    (
        "flatten",
        ["flatten", "my-bot", "BTC/USDT:USDT"],
        {"folder": Path("my-bot"), "symbol": "BTC/USDT:USDT", "external": False},
    ),
    (
        "grid-search",
        ["grid-search", "my-bot", "--output-csv", "scores.csv"],
        {"folder": Path("my-bot"), "output_csv": Path("scores.csv")},
    ),
    (
        "live",
        ["live", "my-bot", "-l", "run.log", "--log-retention-days", "3"],
        {
            "folder": Path("my-bot"),
            "logfile_override": Path("run.log"),
            "console": False,
            "logfiles_retention_days": 3,
        },
    ),
    (
        "report",
        ["report", "my-bot", "--days", "7", "--no-save"],
        {"folder": Path("my-bot"), "days": 7, "save": False},
    ),
    (
        "scheduler",
        ["scheduler", "registry.toml"],
        {"registry": Path("registry.toml")},
    ),
]

DEFAULTS = [
    (
        "backtest",
        {
            "folder": Path("my-bot"),
            "logfile_override": None,
            "console": False,
            "chart": True,
            "save": True,
        },
    ),
    (
        "live",
        {
            "folder": Path("my-bot"),
            "logfile_override": None,
            "console": False,
            "logfiles_retention_days": None,
        },
    ),
    ("report", {"folder": Path("my-bot"), "days": 90, "save": True}),
    ("grid-search", {"folder": Path("my-bot"), "output_csv": None}),
]

BOUNDARY_REPORTS = [
    (
        ExchangeCriticalError("the venue rejected the credentials"),
        "Critical exchange error",
        logging.CRITICAL,
        EX_CONFIG,
    ),
    (
        BotCriticalError("no such strategy"),
        "Critical bot error",
        logging.CRITICAL,
        EX_CONFIG,
    ),
    (
        ExchangeTransientError("the venue timed out"),
        "Venue error",
        logging.ERROR,
        EX_TEMPFAIL,
    ),
    (RuntimeError("unexpected failure"), "Fatal error", logging.ERROR, 1),
]

WORKSPACE_COMMANDS = [
    ("init", ["init", "live"]),
    ("new", ["new", "impulse", "first-bot"]),
    ("check", ["check", "impulse-example"]),
]

AGENT_KINDS = [
    ("agents", Path("agents", "AGENTS.md")),
    ("claude", Path(".claude", "rules", "robottraderslab.md")),
]
AGENT_KIND_NAMES = [kind for kind, _ in AGENT_KINDS]

type ReplaceCommand = Callable[[str, Callable[..., Any]], None]


@pytest.fixture
def replace_command(monkeypatch: pytest.MonkeyPatch) -> ReplaceCommand:
    """Swap the function a command name resolves to for one the test controls."""

    def _replace(name: str, implementation: Callable[..., Any]) -> None:
        module = import_module(COMMAND_SUBPACKAGES[name])
        monkeypatch.setattr(module, "main", implementation)

    return _replace


@pytest.mark.parametrize(("name", "argv", "expected"), DISPATCHES)
def test_command_parses_its_own_arguments(name, argv, expected, cli, replace_command):
    received: list[dict[str, Any]] = []
    replace_command(name, lambda **kwargs: received.append(kwargs))

    invocation = cli.invoke(app, argv)

    assert invocation.exit_code == 0
    assert received == [expected]


@pytest.mark.parametrize(("name", "expected"), DEFAULTS)
def test_command_run_with_the_bot_folder_alone(name, expected, cli, replace_command):
    received: list[dict[str, Any]] = []
    replace_command(name, lambda **kwargs: received.append(kwargs))

    cli.invoke(app, [name, "my-bot"])

    assert received == [expected]


@pytest.mark.parametrize("name", [name for name, _ in DEFAULTS])
def test_command_requires_a_bot_folder(name, cli, replace_command):
    received: list[dict[str, Any]] = []
    replace_command(name, lambda **kwargs: received.append(kwargs))

    invoked = cli.invoke(app, [name])

    assert invoked.exit_code == 2
    assert received == []


def test_init_live_from_a_folder_inside_the_project_fills_its_workspace(
    cli, project, monkeypatch
):
    inside = project / "notes"
    inside.mkdir()
    monkeypatch.chdir(inside)

    cli.invoke(app, ["init", "live"])

    assert (project / "workspace" / "registry.example.toml").is_file()
    assert not (inside / "workspace").exists()


def test_init_live_refuses_a_registry(cli, replace_command):
    received: list[bool] = []
    replace_command("init", lambda workspace: received.append(True))

    invoked = cli.invoke(app, ["init", "live", "--registry", "pilot.toml"])

    assert invoked.exit_code == 2
    assert received == []


@pytest.mark.parametrize(("kind", "instructions"), AGENT_KINDS)
def test_init_for_agents_from_a_folder_inside_the_project_writes_into_the_project(
    kind, instructions, cli, project, monkeypatch
):
    inside = project / "notes"
    inside.mkdir()
    monkeypatch.chdir(inside)

    cli.invoke(app, ["init", kind])

    assert (project / instructions).is_file()
    assert list(inside.iterdir()) == []


@pytest.mark.parametrize("kind", AGENT_KIND_NAMES)
@pytest.mark.parametrize(
    "option", [["--workspace", "elsewhere"], ["--registry", "pilot.toml"]]
)
def test_init_for_agents_refuses_an_option(kind, option, cli, project):
    before = sorted(project.iterdir())

    invoked = cli.invoke(app, ["init", kind, *option])

    assert invoked.exit_code == 2
    assert sorted(project.iterdir()) == before


@pytest.mark.parametrize("kind", AGENT_KIND_NAMES)
def test_init_for_agents_outside_any_project_is_refused(
    kind, cli, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    refused = cli.invoke(app, ["init", kind])

    assert refused.exit_code == 1
    assert "no `pyproject.toml` at or above" in refused.stderr
    assert list(tmp_path.iterdir()) == []


def test_init_requires_a_kind_of_run(cli):
    invoked = cli.invoke(app, ["init"])

    assert invoked.exit_code == 2


def test_init_refuses_a_kind_of_run_it_does_not_lay_out(cli, replace_command):
    received: list[bool] = []
    replace_command("init", lambda workspace: received.append(True))

    invoked = cli.invoke(app, ["init", "backtest"])

    assert invoked.exit_code == 2
    assert received == []


@pytest.mark.usefixtures("project")
def test_new_receives_the_example_and_the_folder(cli, replace_command):
    received: list[tuple[str | None, Path | None]] = []
    replace_command(
        "new", lambda name, folder, workspace: received.append((name, folder))
    )

    cli.invoke(app, ["new", "impulse", "first-bot"])

    assert received == [("impulse", Path("first-bot"))]


@pytest.mark.usefixtures("project")
def test_new_lists_the_installed_examples(cli, replace_command):
    received: list[str | None] = []
    replace_command("new", lambda name, folder, workspace: received.append(name))

    listed = cli.invoke(app, ["new", "--list"])

    assert listed.exit_code == 0
    assert received == [None]


@pytest.mark.parametrize(
    ("name", "argv"), WORKSPACE_COMMANDS, ids=[name for name, _ in WORKSPACE_COMMANDS]
)
def test_a_command_resolves_the_workspace_of_the_project_it_runs_inside(
    name, argv, cli, replace_command, project, monkeypatch
):
    received: list[Path] = []
    replace_command(name, lambda **kwargs: received.append(kwargs["workspace"]))
    inside = project / "notes"
    inside.mkdir()
    monkeypatch.chdir(inside)

    cli.invoke(app, argv)

    assert received == [project / "workspace"]


@pytest.mark.parametrize(
    ("name", "argv"), WORKSPACE_COMMANDS, ids=[name for name, _ in WORKSPACE_COMMANDS]
)
def test_a_command_takes_the_workspace_it_is_given(
    name, argv, cli, replace_command, tmp_path, monkeypatch
):
    received: list[Path] = []
    replace_command(name, lambda **kwargs: received.append(kwargs["workspace"]))
    monkeypatch.chdir(tmp_path)

    given = cli.invoke(app, [*argv, "--workspace", "elsewhere"])

    assert given.exit_code == 0
    assert received == [Path("elsewhere")]


def test_init_cron_refuses_a_workspace(cli, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    invoked = cli.invoke(app, ["init", "cron", "--workspace", "elsewhere"])

    assert invoked.exit_code == 2


@pytest.mark.parametrize(
    "argv",
    [argv for _, argv in WORKSPACE_COMMANDS],
    ids=[name for name, _ in WORKSPACE_COMMANDS],
)
def test_a_command_outside_any_project_is_refused(argv, cli, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    refused = cli.invoke(app, argv)

    assert refused.exit_code == 1
    assert "no `pyproject.toml` at or above" in refused.stderr
    assert list(tmp_path.iterdir()) == []


def test_new_requires_a_name_or_list_but_not_neither(cli):
    invoked = cli.invoke(app, ["new"])

    assert invoked.exit_code == 2


def test_new_requires_a_folder_with_its_name(cli):
    invoked = cli.invoke(app, ["new", "impulse"])

    assert invoked.exit_code == 2


def test_new_requires_a_name_or_list_but_not_both(cli):
    invoked = cli.invoke(app, ["new", "impulse", "--list"])

    assert invoked.exit_code == 2


def test_the_scheduler_reporting_failed_bots(cli, replace_command):
    replace_command("scheduler", lambda **kwargs: 1)

    invoked = cli.invoke(app, ["scheduler", "registry.toml"])

    assert invoked.exit_code == 1


def test_the_scheduler_reporting_no_failed_bots(cli, replace_command):
    replace_command("scheduler", lambda **kwargs: 0)

    invoked = cli.invoke(app, ["scheduler", "registry.toml"])

    assert invoked.exit_code == 0


def test_a_live_instance_that_yielded_to_a_running_one(cli, replace_command):
    replace_command("live", lambda **kwargs: EX_TEMPFAIL)

    invoked = cli.invoke(app, ["live", "my-bot"])

    assert invoked.exit_code == EX_TEMPFAIL


def test_flatten_run_with_only_the_required_arguments(cli, replace_command):
    received: list[dict[str, Any]] = []
    replace_command("flatten", lambda **kwargs: received.append(kwargs))

    cli.invoke(app, ["flatten", "my-bot", "--all"])

    assert received == [{"folder": Path("my-bot"), "symbol": None, "external": False}]


def test_flatten_external_closes_what_no_profile_owns(cli, replace_command):
    received: list[dict[str, Any]] = []
    replace_command("flatten", lambda **kwargs: received.append(kwargs))

    cli.invoke(app, ["flatten", "my-bot", "--external"])

    assert received == [{"folder": Path("my-bot"), "symbol": None, "external": True}]


@pytest.mark.parametrize(
    "argv",
    [
        ["flatten", "my-bot"],
        ["flatten", "my-bot", "BTC/USDT:USDT", "--all"],
        ["flatten", "my-bot", "BTC/USDT:USDT", "--external"],
        ["flatten", "my-bot", "--all", "--external"],
        ["flatten", "my-bot", "BTC/USDT:USDT", "--all", "--external"],
    ],
    ids=["none", "symbol and all", "symbol and external", "all and external", "three"],
)
def test_flatten_takes_exactly_one_of_its_three_forms(argv, cli):
    invoked = cli.invoke(app, argv)

    assert invoked.exit_code == 2
    assert "Give exactly one of SYMBOL, --all or --external" in invoked.stderr


def test_the_flatten_command_reporting_leftovers(cli, replace_command):
    replace_command("flatten", lambda **kwargs: 1)

    invoked = cli.invoke(app, ["flatten", "my-bot", "--all"])

    assert invoked.exit_code == 1


def test_the_flatten_command_reporting_a_flat_account(cli, replace_command):
    replace_command("flatten", lambda **kwargs: 0)

    invoked = cli.invoke(app, ["flatten", "my-bot", "--all"])

    assert invoked.exit_code == 0


def test_help_lists_every_command(cli):
    listing = cli.invoke(app, ["--help"]).output

    assert all(name in listing for name in COMMAND_SUBPACKAGES)


def test_live_help_states_it_places_real_orders(cli):
    help_text = cli.invoke(app, ["live", "--help"]).output

    normalized = " ".join(help_text.split())
    assert (
        "placing real orders on the accounts the bot's `bot.toml` declares"
        in normalized
    )


def test_version_reports_the_installed_distribution(cli):
    reported = cli.invoke(app, ["--version"])

    assert reported.output.strip() == version("robottraderslab")


def test_unknown_command(cli):
    attempted = cli.invoke(app, ["deploy", "my-bot"])

    assert attempted.exit_code == 2


@pytest.mark.parametrize(
    ("failure", "reported", "level", "exit_code"), BOUNDARY_REPORTS
)
def test_what_stops_the_engine_is_recorded(
    failure, reported, level, exit_code, monkeypatch, caplog, replace_command
):
    replace_command("live", _raising(failure))
    monkeypatch.setattr(sys, "argv", ["rtlab", "live", "my-bot"])
    monkeypatch.setattr(logging, "shutdown", lambda: None)

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == exit_code
    assert [record.levelno for record in caplog.records] == [level]
    assert reported in caplog.text
    assert str(failure) in caplog.text


def test_logging_is_flushed_after_a_fatal_error_escapes(monkeypatch, replace_command):
    events: list[str] = []

    def failing(**_: Any) -> None:
        events.append("ran")
        raise RuntimeError("unexpected failure")

    replace_command("live", failing)
    monkeypatch.setattr(sys, "argv", ["rtlab", "live", "my-bot"])
    monkeypatch.setattr(logging, "shutdown", lambda: events.append("flushed"))

    with pytest.raises(SystemExit):
        main()

    assert events == ["ran", "flushed"]


def test_logging_is_flushed_after_a_command_completes(monkeypatch, replace_command):
    events: list[str] = []

    def completing(**_: Any) -> int:
        events.append("ran")
        return 0

    replace_command("live", completing)
    monkeypatch.setattr(sys, "argv", ["rtlab", "live", "my-bot"])
    monkeypatch.setattr(logging, "shutdown", lambda: events.append("flushed"))

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert events == ["ran", "flushed"]


def test_the_terminal_gets_one_sentence_naming_the_cause(
    monkeypatch, capsys, replace_command
):
    failure = BotCriticalError(
        "cannot load the configured notifiers: no plugin named 'discrod'"
    )
    replace_command("live", _raising(failure))
    monkeypatch.setattr(sys, "argv", ["rtlab", "live", "my-bot"])

    with pytest.raises(SystemExit):
        main()

    reported = capsys.readouterr().err
    assert reported.strip() == (
        "Error: cannot load the configured notifiers: no plugin named 'discrod'"
    )


def test_the_terminal_gets_the_traceback_when_asked(
    monkeypatch, capsys, replace_command
):
    replace_command("live", _raising(RuntimeError("unexpected failure")))
    monkeypatch.setattr(sys, "argv", ["rtlab", "--debug", "live", "my-bot"])

    with pytest.raises(SystemExit):
        main()

    assert "Traceback (most recent call last)" in capsys.readouterr().err


def _raising(failure: Exception) -> Callable[..., None]:
    def _raise(**kwargs: Any) -> None:
        raise failure

    return _raise


@pytest.mark.usefixtures("project")
def test_an_example_the_workspace_cannot_be_given(cli, replace_command):
    replace_command("new", _raising(ExampleError("No example named 'ghost'")))

    refused = cli.invoke(app, ["new", "ghost", "first-bot"])

    assert refused.exit_code == 1
    assert "No example named 'ghost'" in refused.stderr


DATA_LIBRARIES = {"numpy", "pandas", "pyarrow"}


@pytest.mark.parametrize("argument", ["--help", "--version"])
def test_a_command_that_touches_no_candle(argument):
    imported_modules = _modules_imported_by(argument)

    assert not DATA_LIBRARIES & imported_modules


LIVE_CONFIG = """
[strategy]
strategy_class = "fake-strategy-99"

[live.trading_account]
account_name = "demoaccount"

[live.ohlcv_provider]
name = "ccxt"
"""


def test_a_second_live_instance_on_a_locked_config(tmp_path):
    config = tmp_path / "bot.toml"
    config.write_text(LIVE_CONFIG)

    with hold_instance_lock(config):
        completed = subprocess.run(
            [sys.executable, "-m", "robottraderslab", "live", str(config.parent)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    assert completed.returncode == EX_TEMPFAIL
    assert f"{tmp_path.name} is already trading in another process" in (
        (tmp_path / "logs" / "live.log").read_text(encoding="utf-8")
    )
    assert (tmp_path / ".rtlab" / "bot.lock").is_file()


def test_a_live_config_that_cannot_be_parsed(tmp_path):
    config = tmp_path / "bot.toml"
    config.write_text("[[bot]\n")

    completed = subprocess.run(
        [sys.executable, "-m", "robottraderslab", "live", str(config.parent)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert completed.returncode == EX_DATAERR
    assert "bot.toml` is not valid TOML" in completed.stderr


def test_a_sweep_named_by_its_file_is_refused_before_any_log(tmp_path):
    sweep = tmp_path / "grid-search.toml"
    sweep.write_text("", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "robottraderslab", "grid-search", str(sweep)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert completed.returncode == EX_DATAERR
    assert f"pass its folder, `{tmp_path}`" in completed.stderr
    assert "Traceback" not in completed.stdout


BACKTEST_CSV_MISSING_KEYS_CONFIG = """
[strategy]
strategy_class = "futures_ma"

[backtest]
maker_fee_rate = 0.0
taker_fee_rate = 0.0
start_date = "2024-01-01"
end_date = "2024-01-02"
initial_balance = { USDT = 1000.0 }

[backtest.ohlcv_provider]
ohlcv_provider = "csv"
file = "empty.csv"
"""


def test_a_backtest_ohlcv_provider_table_missing_its_keys(tmp_path):
    config = tmp_path / "bot.toml"
    config.write_text(BACKTEST_CSV_MISSING_KEYS_CONFIG)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "robottraderslab",
            "backtest",
            str(config.parent),
            "--no-chart",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert completed.returncode == EX_CONFIG
    assert "backtest.ohlcv_provider` needs `symbol`, `timeframe`" in completed.stderr


def test_a_scheduler_pass_over_an_empty_registry(tmp_path):
    registry = tmp_path / "registry.toml"
    registry.write_text("")

    imported_modules = _modules_imported_by("scheduler", str(registry))

    assert not DATA_LIBRARIES & imported_modules


def _modules_imported_by(*arguments: str) -> set[str]:
    completed = subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "robottraderslab", *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return {
        line.rpartition("|")[2].strip()
        for line in completed.stderr.splitlines()
        if line.startswith("import time:")
    }
