import logging
from pathlib import Path

from robottraderslab.live import read_working_copy, working_copy_file


def usable_working_copy(config: Path) -> bytes | None:
    """Return the working copy a bot holds, None when it holds none it can use.

    The copy is the one `config`, the path of the bot's `bot.toml`, names.
    Every decision the scheduler takes on a bot's copy reads it here, so that
    they agree on which bots hold one. A copy that cannot be read or is not
    valid TOML is logged at DEBUG only, since the scheduler reads every
    registered bot's copy on every run. A bot that holds a copy it cannot use
    is either launched, and its own cycle reports the copy, or skipped when
    the registry is read, with a line naming the copy.
    """
    return read_working_copy(working_copy_file(config), logging.DEBUG)


def holds_usable_copy(config: Path) -> bool:
    """Return whether the bot of the `bot.toml` at `config` holds a usable copy."""
    return usable_working_copy(config) is not None
