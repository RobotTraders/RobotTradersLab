import logging
import subprocess
import sys
from collections.abc import Callable
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from robottraderslab._core import (
    EX_CONFIG,
    EX_DATAERR,
    EX_TEMPFAIL,
    hold_instance_lock,
)
from robottraderslab.cli import app, main
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeTransientError,
    StrategyCriticalError,
)
from robottraderslab.scaffold import ExampleError

COMMAND_SUBPACKAGES = {
    "backtest": "robottraderslab.backtester",
    "flatten": "robottraderslab.live.flatten",
    "grid-search": "robottraderslab.grid_search",
    "live": "robottraderslab.live",
    "report": "robottraderslab.live.report",
    "scaffold": "robottraderslab.scaffold",
    "scheduler": "robottraderslab.scheduler",
}

DISPATCHES = [
    (
        "backtest",
        [
            "backtest",
            "cfg.toml",
            "--log-file",
            "run.log",
            "--console",
            "--no-chart",
            "--no-save",
        ],
        {
            "config": Path("cfg.toml"),
            "logfile_override": Path("run.log"),
            "console": True,
            "chart": False,
            "save": False,
        },
    ),
    (
        "flatten",
        ["flatten", "cfg.toml", "BTC/USDT:USDT"],
        {"config": Path("cfg.toml"), "symbol": "BTC/USDT:USDT", "external": False},
    ),
    (
        "grid-search",
        ["grid-search", "cfg.toml", "--output-csv", "scores.csv"],
        {"config": Path("cfg.toml"), "output_csv": Path("scores.csv")},
    ),
    (
        "live",
        ["live", "cfg.toml", "-l", "run.log", "--log-retention-days", "3"],
        {
            "config": Path("cfg.toml"),
            "logfile_override": Path("run.log"),
            "console": False,
            "logfiles_retention_days": 3,
        },
    ),
    (
        "report",
        ["report", "cfg.toml", "--days", "7"],
        {"config": Path("cfg.toml"), "days": 7},
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
            "config": Path("bot-example.toml"),
            "logfile_override": None,
            "console": False,
            "chart": True,
            "save": True,
        },
    ),
    (
        "live",
        {
            "config": Path("bot-example.toml"),
            "logfile_override": None,
            "console": False,
            "logfiles_retention_days": None,
        },
    ),
    ("report", {"config": Path("bot-example.toml"), "days": 90}),
    ("grid-search", {"config": Path("bot-example.toml"), "output_csv": None}),
]

BOUNDARY_REPORTS = [
    (
        ExchangeCriticalError("the venue rejected the credentials"),
        "Critical exchange error",
        EX_CONFIG,
    ),
    (StrategyCriticalError("no such strategy"), "Critical strategy error", EX_CONFIG),
    (ExchangeTransientError("the venue timed out"), "Venue error", EX_TEMPFAIL),
    (RuntimeError("unexpected failure"), "Fatal error", 1),
]

type ReplaceCommand = Callable[[str, Callable[..., Any]], None]


@pytest.fixture
def cli() -> CliRunner:
    return CliRunner()


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
def test_command_run_with_the_config_alone(name, expected, cli, replace_command):
    received: list[dict[str, Any]] = []
    replace_command(name, lambda **kwargs: received.append(kwargs))

    cli.invoke(app, [name, "bot-example.toml"])

    assert received == [expected]


@pytest.mark.parametrize("name", [name for name, _ in DEFAULTS])
def test_command_requires_a_config(name, cli, replace_command):
    received: list[dict[str, Any]] = []
    replace_command(name, lambda **kwargs: received.append(kwargs))

    invoked = cli.invoke(app, [name])

    assert invoked.exit_code == 2
    assert received == []


def test_scaffold_receives_the_example_name(cli, replace_command):
    received: list[str | None] = []
    replace_command("scaffold", lambda name: received.append(name))

    cli.invoke(app, ["scaffold", "impulse"])

    assert received == ["impulse"]


def test_scaffold_lists_the_installed_examples(cli, replace_command):
    received: list[str | None] = []
    replace_command("scaffold", lambda name: received.append(name))

    listed = cli.invoke(app, ["scaffold", "--list"])

    assert listed.exit_code == 0
    assert received == [None]


def test_scaffold_requires_a_name_or_list_but_not_neither(cli):
    invoked = cli.invoke(app, ["scaffold"])

    assert invoked.exit_code == 2


def test_scaffold_requires_a_name_or_list_but_not_both(cli):
    invoked = cli.invoke(app, ["scaffold", "impulse", "--list"])

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

    invoked = cli.invoke(app, ["live", "cfg.toml"])

    assert invoked.exit_code == EX_TEMPFAIL


def test_flatten_run_with_only_the_required_arguments(cli, replace_command):
    received: list[dict[str, Any]] = []
    replace_command("flatten", lambda **kwargs: received.append(kwargs))

    cli.invoke(app, ["flatten", "cfg.toml", "--all"])

    assert received == [{"config": Path("cfg.toml"), "symbol": None, "external": False}]


def test_flatten_external_closes_what_no_profile_owns(cli, replace_command):
    received: list[dict[str, Any]] = []
    replace_command("flatten", lambda **kwargs: received.append(kwargs))

    cli.invoke(app, ["flatten", "cfg.toml", "--external"])

    assert received == [{"config": Path("cfg.toml"), "symbol": None, "external": True}]


@pytest.mark.parametrize(
    "argv",
    [
        ["flatten", "cfg.toml"],
        ["flatten", "cfg.toml", "BTC/USDT:USDT", "--all"],
        ["flatten", "cfg.toml", "BTC/USDT:USDT", "--external"],
        ["flatten", "cfg.toml", "--all", "--external"],
        ["flatten", "cfg.toml", "BTC/USDT:USDT", "--all", "--external"],
    ],
    ids=["none", "symbol and all", "symbol and external", "all and external", "three"],
)
def test_flatten_takes_exactly_one_of_its_three_forms(argv, cli):
    invoked = cli.invoke(app, argv)

    assert invoked.exit_code == 2
    assert "Give exactly one of SYMBOL, --all or --external" in invoked.stderr


def test_the_flatten_command_reporting_leftovers(cli, replace_command):
    replace_command("flatten", lambda **kwargs: 1)

    invoked = cli.invoke(app, ["flatten", "cfg.toml", "--all"])

    assert invoked.exit_code == 1


def test_the_flatten_command_reporting_a_flat_account(cli, replace_command):
    replace_command("flatten", lambda **kwargs: 0)

    invoked = cli.invoke(app, ["flatten", "cfg.toml", "--all"])

    assert invoked.exit_code == 0


def test_help_lists_every_command(cli):
    listing = cli.invoke(app, ["--help"]).output

    assert all(name in listing for name in COMMAND_SUBPACKAGES)


def test_live_help_states_it_places_real_orders(cli):
    help_text = cli.invoke(app, ["live", "--help"]).output

    normalized = " ".join(help_text.split())
    assert (
        "placing real orders on the accounts the configuration file declares"
        in normalized
    )


def test_version_reports_the_installed_distribution(cli):
    reported = cli.invoke(app, ["--version"])

    assert reported.output.strip() == version("robottraderslab")


def test_unknown_command(cli):
    attempted = cli.invoke(app, ["deploy", "cfg.toml"])

    assert attempted.exit_code == 2


@pytest.mark.parametrize(("failure", "reported", "exit_code"), BOUNDARY_REPORTS)
def test_what_stops_the_engine_is_recorded(
    failure, reported, exit_code, monkeypatch, caplog, replace_command
):
    replace_command("live", _raising(failure))
    monkeypatch.setattr(sys, "argv", ["rtlab", "live", "cfg.toml"])
    monkeypatch.setattr(logging, "shutdown", lambda: None)

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == exit_code
    assert reported in caplog.text
    assert str(failure) in caplog.text


def test_logging_is_flushed_before_a_fatal_error_escapes(monkeypatch, replace_command):
    replace_command("live", _raising(RuntimeError("unexpected failure")))
    monkeypatch.setattr(sys, "argv", ["rtlab", "live", "cfg.toml"])
    flushes: list[None] = []
    monkeypatch.setattr(logging, "shutdown", lambda: flushes.append(None))

    with pytest.raises(SystemExit):
        main()

    assert flushes == [None]


def test_the_terminal_gets_one_sentence_naming_the_cause(
    monkeypatch, capsys, replace_command
):
    failure = StrategyCriticalError(
        "cannot load the configured notifiers: no plugin named 'discrod'"
    )
    replace_command("live", _raising(failure))
    monkeypatch.setattr(sys, "argv", ["rtlab", "live", "cfg.toml"])

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
    monkeypatch.setattr(sys, "argv", ["rtlab", "--debug", "live", "cfg.toml"])

    with pytest.raises(SystemExit):
        main()

    assert "Traceback (most recent call last)" in capsys.readouterr().err


def _raising(failure: Exception) -> Callable[..., None]:
    def _raise(**kwargs: Any) -> None:
        raise failure

    return _raise


def test_an_example_the_workspace_cannot_be_given(cli, replace_command):
    replace_command("scaffold", _raising(ExampleError("No example named 'ghost'")))

    refused = cli.invoke(app, ["scaffold", "ghost"])

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
            [sys.executable, "-m", "robottraderslab", "live", str(config)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    assert completed.returncode == EX_TEMPFAIL
    assert "bot.toml is already trading in another process" in (
        (tmp_path / "logs" / "live_bot.log").read_text(encoding="utf-8")
    )
    assert (tmp_path / ".rtlab" / "bot.lock").is_file()


def test_a_live_config_that_cannot_be_parsed(tmp_path):
    config = tmp_path / "bot.toml"
    config.write_text("[[bot]\n")

    completed = subprocess.run(
        [sys.executable, "-m", "robottraderslab", "live", str(config)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert completed.returncode == EX_DATAERR
    assert "bot.toml` is not valid TOML" in completed.stderr


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
            str(config),
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
