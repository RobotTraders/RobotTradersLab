import logging
import traceback
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

import typer

from robottraderslab._core import EX_CONFIG, EX_DATAERR, EX_TEMPFAIL
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
    StrategyCriticalError,
)
from robottraderslab.scaffold import ExampleError

from .console import use_utf8_console

logger = logging.getLogger(__name__)

_show_traceback = False

_DISTRIBUTION = "robottraderslab"
_DEFAULT_REPORT_DAYS = 90
_EX_CRASH = 1

_SUBPACKAGES = {
    "backtest": "robottraderslab.backtester",
    "flatten": "robottraderslab.live.flatten",
    "grid-search": "robottraderslab.grid_search",
    "live": "robottraderslab.live",
    "report": "robottraderslab.live.report",
    "scaffold": "robottraderslab.scaffold",
    "scheduler": "robottraderslab.scheduler",
}

_ConfigFile = Annotated[
    Path,
    typer.Argument(
        metavar="CONFIG",
        help="The bot's configuration file",
    ),
]
_LogFile = Annotated[
    Path | None,
    typer.Option(
        "--log-file",
        "-l",
        help="Where to write this run's log, overriding the configuration file",
    ),
]
_ConsoleFlag = Annotated[
    bool, typer.Option("--console", help="Mirror the log to the console")
]
_DebugFlag = Annotated[
    bool,
    typer.Option("--debug", help="Print the traceback of a failure as well"),
]
_Symbol = Annotated[
    str | None,
    typer.Argument(
        metavar="SYMBOL",
        help="Symbol to flatten, such as `BTC/USDT:USDT`; omit and pass --all to "
        "flatten every symbol carrying an open order or position",
    ),
]
_AllSymbolsFlag = Annotated[
    bool,
    typer.Option(
        "--all", help="Flatten every symbol carrying an open order or position"
    ),
]
_ExternalFlag = Annotated[
    bool,
    typer.Option(
        "--external",
        help="Close only what no profile of the bot owns, on every market its "
        "profiles declare, and leave the bot trading",
    ),
]
_ExampleName = Annotated[
    str | None,
    typer.Argument(
        metavar="NAME", help="Example to copy; pass --list to see the installed ones"
    ),
]
_ListExamplesFlag = Annotated[
    bool,
    typer.Option("--list", help="List the examples the installed plugins ship"),
]

app = typer.Typer(
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def main() -> None:
    use_utf8_console()
    try:
        app()
    except ExchangeCriticalError as e:
        _report(logging.CRITICAL, "Critical exchange error", e, EX_CONFIG)
    except StrategyCriticalError as e:
        _report(logging.CRITICAL, "Critical strategy error", e, EX_CONFIG)
    except ExchangeRecoverableError as e:
        _report(logging.ERROR, "Venue error", e, EX_TEMPFAIL)
    except Exception as e:
        _report(logging.ERROR, "Fatal error", e, _EX_CRASH)


@app.callback(invoke_without_command=True)
def root(
    show_version: Annotated[
        bool,
        typer.Option("--version", help="Show the installed version and exit"),
    ] = False,
    debug: _DebugFlag = False,
) -> None:
    """Command line of the RobotTradersLab engine: backtest a bot, search its
    strategy's parameters, run it live, report on its account and flatten it.
    """
    _set_traceback(debug)
    if show_version:
        typer.echo(version(_DISTRIBUTION))
        raise typer.Exit()


@app.command()
def backtest(
    config: _ConfigFile,
    logfile_override: _LogFile = None,
    console: _ConsoleFlag = False,
    chart: Annotated[
        bool,
        typer.Option(
            "--chart/--no-chart",
            help="Open the interactive report in the browser at the end",
        ),
    ] = True,
    save: Annotated[
        bool,
        typer.Option(
            "--save/--no-save",
            help="Keep the run under reports/ beside the bot: its "
            "configuration file, the printed report and the HTML report",
        ),
    ] = True,
) -> None:
    """Replay a bot over past candles and report how it would have performed."""
    _run(
        "backtest",
        config=config,
        logfile_override=logfile_override,
        console=console,
        chart=chart,
        save=save,
    )


@app.command(
    epilog="Examples: `rtlab flatten workspace/first-bot/impulse-bot-example.toml "
    "BTC/USDT:USDT` flattens one symbol; `rtlab flatten "
    "workspace/first-bot/impulse-bot-example.toml --all` flattens every symbol "
    "the account carries; `rtlab flatten "
    "workspace/first-bot/impulse-bot-example.toml --external` reads ninety days "
    "of fills, gives each to the profile whose tag it carries and closes the "
    "difference on each market, open orders untouched. A market a hand holds on "
    "the other side of the bot's own position is left alone and named, since "
    "closing it would reset every profile; flatten that market whole instead. "
    "The command exits 1 while any market still holds what no profile owns."
)
def flatten(
    ctx: typer.Context,
    config: _ConfigFile,
    symbol: _Symbol = None,
    all_symbols: _AllSymbolsFlag = False,
    external: _ExternalFlag = False,
) -> None:
    """Flatten the account a bot trades, cancelling every open order and closing
    every open position, or close the quantity no profile of the bot owns.
    """
    if [symbol is not None, all_symbols, external].count(True) != 1:
        ctx.fail("Give exactly one of SYMBOL, --all or --external")
    if _run("flatten", config=config, symbol=symbol, external=external):
        raise typer.Exit(1)


@app.command("grid-search")
def grid_search(
    config: _ConfigFile,
    output_csv: Annotated[
        Path | None, typer.Option(help="Where to write the scored combinations")
    ] = None,
) -> None:
    """Backtest every parameter combination the configuration file's
    `[optimisation]` section declares, and score each.
    """
    _run("grid-search", config=config, output_csv=output_csv)


@app.command()
def live(
    config: _ConfigFile,
    logfile_override: _LogFile = None,
    console: _ConsoleFlag = False,
    logfiles_retention_days: Annotated[
        int | None,
        typer.Option(
            "--log-retention-days",
            help="How many days of log files to keep, overriding the configuration file",
        ),
    ] = None,
) -> None:
    """Trade one cycle on the candles that have just closed, placing real
    orders on the accounts the configuration file declares.

    One instance of a bot trades at a time, so a second `rtlab live` on the
    same configuration file logs one warning and exits without touching the
    venue.
    """
    exit_code = _run(
        "live",
        config=config,
        logfile_override=logfile_override,
        console=console,
        logfiles_retention_days=logfiles_retention_days,
    )
    if exit_code:
        raise typer.Exit(exit_code)


@app.command()
def report(
    config: _ConfigFile,
    days: Annotated[
        int, typer.Option(help="How many days back the report covers")
    ] = _DEFAULT_REPORT_DAYS,
) -> None:
    """Open the interactive report of a live account, its trades drawn over the
    strategy's indicators.
    """
    _run("report", config=config, days=days)


@app.command(
    epilog="Examples: `rtlab scaffold --list` names the examples the installed plugins "
    "ship; `rtlab scaffold impulse` copies that one into "
    "`workspace/impulse-bot-example/`."
)
def scaffold(
    ctx: typer.Context,
    name: _ExampleName = None,
    list_examples: _ListExamplesFlag = False,
) -> None:
    """Copy a packaged example into the workspace, or list the installed ones."""
    if (name is not None) == list_examples:
        ctx.fail("Give exactly one of NAME or --list")
    try:
        _run("scaffold", name=name)
    except ExampleError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e


@app.command()
def scheduler(
    registry: Annotated[
        Path,
        typer.Argument(
            metavar="REGISTRY",
            help="Registry file listing the bots and their timeframes",
        ),
    ],
) -> None:
    """Launch every registered bot whose candle closes this minute.

    One run works a registry at a time, so a run starting while the previous
    one is still going launches no bot and logs one warning.
    """
    if _run("scheduler", registry=registry):
        raise typer.Exit(1)


def _set_traceback(debug: bool) -> None:
    global _show_traceback
    _show_traceback = debug


def _run(command: str, **arguments: Any) -> int | None:
    failures: int | None = import_module(_SUBPACKAGES[command]).main(**arguments)
    return failures


def _report(
    level: int, headline: str, failure: BaseException, spoken_code: int
) -> None:
    """A failure raised before the log exists reaches nobody but the terminal, so
    it exits on a code of its own and whoever launched the run carries that
    sentence on its behalf.
    """
    exit_code = spoken_code
    if logging.getLogger().hasHandlers():
        logger.log(level, "%s: %s", headline, failure, exc_info=True)
        logging.shutdown()
    else:
        exit_code = EX_DATAERR
    typer.echo(f"Error: {failure}", err=True)
    if _show_traceback:
        traceback.print_exception(failure)
    raise SystemExit(exit_code)
