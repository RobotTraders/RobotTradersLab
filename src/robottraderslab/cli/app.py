import logging
import traceback
from enum import StrEnum
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from robottraderslab._core import EX_CONFIG, EX_DATAERR, EX_TEMPFAIL
from robottraderslab.exceptions import (
    BotCriticalError,
    ExchangeCriticalError,
    ExchangeRecoverableError,
)
from robottraderslab.scaffold import (
    CrontabError,
    ExampleError,
    OutsideProjectError,
    project_folder,
)

from .console import use_utf8_console

logger = logging.getLogger(__name__)

_show_traceback = False

_DISTRIBUTION = "robottraderslab"
_DEFAULT_REPORT_DAYS = 90
_EX_CRASH = 1

_SUBPACKAGES = {
    "backtest": "robottraderslab.backtester",
    "check": "robottraderslab.live.check",
    "flatten": "robottraderslab.live.flatten",
    "grid-search": "robottraderslab.grid_search",
    "live": "robottraderslab.live",
    "report": "robottraderslab.live.report",
    "new": "robottraderslab.scaffold",
    "scheduler": "robottraderslab.scheduler",
}


class _InitKind(StrEnum):
    AGENTS = "agents"
    CLAUDE = "claude"
    CRON = "cron"
    LIVE = "live"


_INIT_SUBPACKAGES = {
    _InitKind.AGENTS: "robottraderslab.scaffold.agents",
    _InitKind.CLAUDE: "robottraderslab.scaffold.claude",
    _InitKind.CRON: "robottraderslab.scaffold.cron",
    _InitKind.LIVE: "robottraderslab.scaffold.workspace",
}


_BotFolder = Annotated[
    Path,
    typer.Argument(
        metavar="BOT",
        help="The bot's folder, holding its `bot.toml`; a path to a `.toml` "
        "file is refused, naming the folder to pass",
    ),
]
_LogFile = Annotated[
    Path | None,
    typer.Option(
        "--log-file",
        "-l",
        help="Where to write this run's log, overriding `bot.toml`",
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
_ExampleFolder = Annotated[
    Path | None,
    typer.Argument(
        metavar="FOLDER",
        help="The bot's folder to create in the workspace, its name the bot's",
    ),
]
_ListExamplesFlag = Annotated[
    bool,
    typer.Option("--list", help="List the examples the installed plugins ship"),
]


_InitKindArgument = Annotated[
    _InitKind,
    typer.Argument(metavar="KIND", help="`live`, `cron`, `claude` or `agents`"),
]
_Registry = Annotated[
    Path | None,
    typer.Option(
        "--registry",
        help="The registry the cron line runs the scheduler on, "
        "`workspace/registry.toml` unless given",
    ),
]
_WORKSPACE = "workspace"
_DEFAULT_REGISTRY = Path(_WORKSPACE) / "registry.toml"
_SecretsEntry = Annotated[
    str, typer.Argument(metavar="ENTRY", help="Name of the `[[secrets]]` entry")
]
_SecretsFile = Annotated[
    Path,
    typer.Option("--secrets", help="The secrets file, relative to the workspace"),
]
_AccountSymbol = Annotated[
    str,
    typer.Option(
        "--symbol",
        help="The market whose account a venue key is read on, for a venue keeping "
        "one account per settlement currency: on Bitget, `BTC/USDT:USDT` reads "
        "the USDT-settled futures account and `ETH/USDC:USDC` the USDC-settled one",
    ),
]
_DEFAULT_SECRETS_FILE = Path("secrets.toml")
_DEFAULT_ACCOUNT_SYMBOL = "BTC/USDT:USDT"
_Workspace = Annotated[
    Path | None,
    typer.Option(
        "--workspace",
        help="The workspace folder, `workspace/` under the project unless given",
    ),
]

app = typer.Typer(
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def main() -> None:
    """A handler delivering over the network needs the thread pool the
    interpreter retires before it runs the logging shutdown it registers.
    """
    use_utf8_console()
    try:
        app()
    except ExchangeCriticalError as e:
        _report(logging.CRITICAL, "Critical exchange error", e, EX_CONFIG)
    except BotCriticalError as e:
        _report(logging.CRITICAL, "Critical bot error", e, EX_CONFIG)
    except ExchangeRecoverableError as e:
        _report(logging.ERROR, "Venue error", e, EX_TEMPFAIL)
    except Exception as e:
        _report(logging.ERROR, "Fatal error", e, _EX_CRASH)
    finally:
        logging.shutdown()


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
    bot: _BotFolder,
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
            help="Keep the run under `backtests/<time>/` in the bot's folder: "
            "its `bot.toml`, its `backtest.log` unless a log file is named, "
            "`report.txt` and `report.html`; --no-save writes no run folder "
            "and shows the log in the terminal unless a log file is named",
        ),
    ] = True,
) -> None:
    """Replay a bot over past candles and report how it would have performed."""
    _run(
        "backtest",
        folder=bot,
        logfile_override=logfile_override,
        console=console,
        chart=chart,
        save=save,
    )


@app.command(
    epilog="Examples: `rtlab check impulse-example` reads the balances of the "
    "account the `impulse-example` key opens; `rtlab check discord-trades` posts "
    "one message to the `discord-trades` webhook. A connector also reports the "
    "account settings it trades under. The command exits 1 on every failure it "
    "names: a refused key, a refused or unreachable webhook, a missing file, "
    "entry, connector or plugin, an entry naming neither `exchange` nor "
    "`webhook_url`, an account setting the connector does not trade under, a "
    "symbol the venue does not list."
)
def check(
    entry: _SecretsEntry,
    secrets: _SecretsFile = _DEFAULT_SECRETS_FILE,
    symbol: _AccountSymbol = _DEFAULT_ACCOUNT_SYMBOL,
    workspace: _Workspace = None,
) -> None:
    """Prove a `[[secrets]]` entry of the workspace's secrets file: a venue key
    by reading the balances of the account it opens, a webhook by posting one
    message to it.
    """
    folder = _workspace(workspace)
    try:
        _run("check", entry=entry, secrets=secrets, symbol=symbol, workspace=folder)
    except (Exception, ExchangeCriticalError, BotCriticalError) as e:
        _refuse(e)


@app.command(
    epilog="Examples: `rtlab flatten workspace/first-bot BTC/USDT:USDT` flattens "
    "one symbol; `rtlab flatten workspace/first-bot --all` flattens every symbol "
    "the account carries; `rtlab flatten workspace/first-bot --external` reads "
    "ninety days "
    "of fills, gives each to the profile whose tag it carries and closes the "
    "difference on each market, open orders untouched. A market a hand holds on "
    "the other side of the bot's own position is left alone and named, since "
    "closing it would reset every profile; flatten that market whole instead. "
    "The command exits 1 while any market still holds what no profile owns."
)
def flatten(
    ctx: typer.Context,
    bot: _BotFolder,
    symbol: _Symbol = None,
    all_symbols: _AllSymbolsFlag = False,
    external: _ExternalFlag = False,
) -> None:
    """Flatten the account a bot trades, cancelling every open order and closing
    every open position, or close the quantity no profile of the bot owns.
    """
    if [symbol is not None, all_symbols, external].count(True) != 1:
        ctx.fail("Give exactly one of SYMBOL, --all or --external")
    if _run("flatten", folder=bot, symbol=symbol, external=external):
        raise typer.Exit(1)


@app.command("grid-search")
def grid_search(
    bot: _BotFolder,
    output_csv: Annotated[
        Path | None,
        typer.Option(
            help="Also write the scored combinations to this file, beside the "
            "`results.csv` the run folder holds"
        ),
    ] = None,
) -> None:
    """Backtest every parameter combination the bot's `grid-search.toml`
    declares, and score each, in a `grid-searches/<time>/` folder of the bot
    holding the two files it ran, its log and `results.csv`.
    """
    _run("grid-search", folder=bot, output_csv=output_csv)


@app.command()
def init(
    ctx: typer.Context,
    kind: _InitKindArgument,
    registry: _Registry = None,
    workspace: _Workspace = None,
) -> None:
    """Set up a project for live trading or for an AI agent: `live` lays out
    the example secrets file and the example registry of bots in the workspace,
    to copy and fill; `cron` makes sure the crontab of the user running it holds
    the line that runs the project's scheduler every minute; `claude` sets up
    Claude with the engine's agent instructions, `AGENTS.md`, and its skills,
    the instructions as `.claude/rules/robottraderslab.md` and the skills under
    `.claude/skills/`, leaving `CLAUDE.md` untouched; `agents` writes the same
    files into `agents/`, for any other agent to copy and rename as it expects.
    A rerun of `claude` or `agents` replaces the copies with the ones the
    installed engine ships.
    """
    if kind in (_InitKind.AGENTS, _InitKind.CLAUDE):
        if workspace is not None or registry is not None:
            ctx.fail(f"`init {kind}` takes no option")
        _init(kind, project=_project())
        return
    if kind is _InitKind.CRON:
        if workspace is not None:
            ctx.fail("--workspace goes with `init live`")
        project = _project()
        try:
            _init(kind, project=project, registry=registry or _DEFAULT_REGISTRY)
        except CrontabError as e:
            _refuse(e)
        return
    if registry is not None:
        ctx.fail("--registry goes with `init cron`")
    _init(kind, workspace=_workspace(workspace))


@app.command()
def live(
    bot: _BotFolder,
    logfile_override: _LogFile = None,
    console: _ConsoleFlag = False,
    logfiles_retention_days: Annotated[
        int | None,
        typer.Option(
            "--log-retention-days",
            help="How many days of log files to keep, overriding `bot.toml`",
        ),
    ] = None,
) -> None:
    """Trade one cycle on the candles that have just closed, placing real
    orders on the accounts the bot's `bot.toml` declares, and log to the
    bot's `logs/live.log`.

    One instance of a bot trades at a time, so a second `rtlab live` on the
    same bot logs one warning and exits without touching the venue.
    """
    exit_code = _run(
        "live",
        folder=bot,
        logfile_override=logfile_override,
        console=console,
        logfiles_retention_days=logfiles_retention_days,
    )
    if exit_code:
        raise typer.Exit(exit_code)


@app.command()
def report(
    bot: _BotFolder,
    days: Annotated[
        int, typer.Option(help="How many days back the report covers")
    ] = _DEFAULT_REPORT_DAYS,
    save: Annotated[
        bool,
        typer.Option(
            "--save/--no-save",
            help="Keep the report as `reports/<time>/report.html` in the bot's "
            "folder; --no-save only opens it",
        ),
    ] = True,
) -> None:
    """Open the interactive report of a live account, its trades drawn over the
    strategy's indicators.
    """
    _run("report", folder=bot, days=days, save=save)


@app.command(
    epilog="Examples: `rtlab new --list` names the examples the installed plugins "
    "ship; `rtlab new impulse first-bot` copies that one into "
    "`workspace/first-bot/`, the bot's folder holding its `bot.toml`."
)
def new(
    ctx: typer.Context,
    name: _ExampleName = None,
    folder: _ExampleFolder = None,
    list_examples: _ListExamplesFlag = False,
    workspace: _Workspace = None,
) -> None:
    """Copy a packaged example into the workspace as a bot's folder, or list
    the installed ones.
    """
    if list_examples:
        if name is not None or folder is not None:
            ctx.fail("--list takes no NAME or FOLDER")
    elif name is None or folder is None:
        ctx.fail("Give NAME and FOLDER, or --list")
    try:
        _run("new", name=name, folder=folder, workspace=_workspace(workspace))
    except ExampleError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e


@app.command()
def scheduler(
    registry: Annotated[
        Path,
        typer.Argument(
            metavar="REGISTRY",
            help="Registry file listing the bots' folders and their timeframes",
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
    return _call(_SUBPACKAGES[command], **arguments)


def _init(kind: _InitKind, **arguments: Any) -> None:
    _call(_INIT_SUBPACKAGES[kind], **arguments)


def _call(subpackage: str, **arguments: Any) -> int | None:
    failures: int | None = import_module(subpackage).main(**arguments)
    return failures


def _refuse(failure: BaseException) -> NoReturn:
    """A failure a command reports before any log exists is its whole outcome,
    so it exits 1 whatever its class.
    """
    typer.echo(f"Error: {failure}", err=True)
    if _show_traceback:
        traceback.print_exception(failure)
    raise typer.Exit(_EX_CRASH) from failure


def _workspace(given: Path | None) -> Path:
    return _project() / _WORKSPACE if given is None else given


def _project() -> Path:
    try:
        return project_folder(Path.cwd())
    except OutsideProjectError as e:
        _refuse(e)


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
    else:
        exit_code = EX_DATAERR
    typer.echo(f"Error: {failure}", err=True)
    if _show_traceback:
        traceback.print_exception(failure)
    raise SystemExit(exit_code)
