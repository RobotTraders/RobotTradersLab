import logging
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from robottraderslab._internal import DryRunFailure, handlers_on_root
from robottraderslab.bootstrap import (
    BotConfig,
    bot_name,
    expected_bot_file,
    names_a_file,
)
from robottraderslab_core.engine import name_process

from .runners import contained_dry_run, dry_run_livebot
from .working_copy import load_bot_file

_NOT_CHECKED = "Not checked, run it again: "


def main(folder: Path, draft_file: Path | None) -> int:
    """Check a bot's configuration file with the dry run of a live cycle.

    The file is the bot's `bot.toml`, or the draft `draft_file` names, read as
    that bot's `bot.toml`, so its relative paths, its secrets file and the bot's
    name are the bot's. A relative `draft_file` starts from the bot's folder.
    Every symbol counts. The command reads the venue with the bot's keys and
    sends nothing. It takes no instance lock and writes no log, working copy or
    `bot.toml`; a candle source that keeps candles stores the ones it fetches in
    the bot's candle store, as every run does. It prints one line on stderr for
    each failure and, when there is none, one line on stdout saying the file
    passes.

    Args:
        folder: The bot's folder, holding its `bot.toml`.
        draft_file: The file to check in place of `bot.toml`, None to check
            `bot.toml`.

    Returns:
        1 when the dry run met any failure, a read that failed included, else 0.
    """
    config = expected_bot_file(folder)
    draft = None if draft_file is None else folder / draft_file
    folder_name = _folder_name(folder)
    with handlers_on_root([logging.NullHandler()]):
        failures = _failures(folder, draft)
    for failure in failures:
        print(_line(failure, folder_name, config, draft), file=sys.stderr)
    if not failures:
        print(f"`{draft or config}` passes the dry run")
    return 1 if failures else 0


def _folder_name(folder: Path) -> str:
    """The name of the bot the argument names, or of the folder a refusal names."""
    return bot_name(
        expected_bot_file(folder.parent if names_a_file(folder) else folder)
    )


def _failures(folder: Path, draft: Path | None) -> list[DryRunFailure]:
    return contained_dry_run(partial(_check, folder, draft, datetime.now(UTC)))


def _check(folder: Path, draft: Path | None, moment: datetime) -> list[DryRunFailure]:
    bot_config, _ = load_bot_file(folder, draft)
    name_process(bot_config.name, moment)
    return _missing_secrets_file(bot_config) or dry_run_livebot(bot_config, moment)


def _missing_secrets_file(bot_config: BotConfig) -> list[DryRunFailure]:
    """The failure of a secrets file the bot declares that does not exist.

    The dry run reads a missing file as one with no entry and names the entry
    its account lacks, never the file, so this failure stands alone and the dry
    run does not run.
    """
    secrets_file = bot_config.secrets_file
    if secrets_file is None or Path(secrets_file).exists():
        return []
    message = f"The secrets file `{secrets_file}` does not exist"
    return [DryRunFailure(None, message, recoverable=False)]


def _line(
    failure: DryRunFailure, folder_name: str, config: Path, draft: Path | None
) -> str:
    where = str(failure.symbol) if failure.symbol is not None else folder_name
    message = failure.message
    if draft is not None and f"`{config}`" in message:
        message = f"`{draft}`, loaded as `bot.toml`: {message}"
    prefix = _NOT_CHECKED if failure.recoverable else ""
    return f"{prefix}{where}: {message}"
