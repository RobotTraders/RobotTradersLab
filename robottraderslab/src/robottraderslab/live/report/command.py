import logging
from pathlib import Path

from ..working_copy import load_running_config, naming_the_copy
from .runners import run_report


def main(folder: Path, days: int, save: bool) -> None:
    """Chart the account's performance over the window against its indicators.

    The account is the one of the bot's working copy when it can be read and is
    valid TOML, and of its `bot.toml` otherwise. The command writes no working copy.

    Raises:
        BotCriticalError: If the working copy the bot holds, or its `bot.toml`
            when it holds no usable copy, declares no bot the engine can run,
            no `[live]` section or no profile. The error names the working copy
            when it is the file read.
        ExchangeCriticalError: If the account cannot be read.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    running = load_running_config(folder)
    with naming_the_copy(running):
        run_report(running.bot, running.text, days=days, save=save)
