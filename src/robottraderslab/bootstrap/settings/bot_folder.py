from pathlib import Path

from robottraderslab.exceptions import BotCriticalError

_BOT_FILE_NAME = "bot.toml"
_OPTIMISATION_FILE_NAME = "optimisation.toml"
_TOML_SUFFIX = ".toml"


def bot_file(folder: Path) -> Path:
    """A bot is the folder holding its `bot.toml`, so every command names the
    folder and finds the file under that fixed name.

    Raises:
        BotCriticalError: If the path is a file, or the folder holds no
            `bot.toml`.
    """
    return _file_in(folder, _BOT_FILE_NAME)


def optimisation_file(folder: Path) -> Path:
    """A bot's parameter sweep sits beside its `bot.toml` under a fixed name.

    Raises:
        BotCriticalError: If the path is a file, or the folder holds no
            `optimisation.toml`.
    """
    return _file_in(folder, _OPTIMISATION_FILE_NAME)


def bot_name(config_file: Path) -> str:
    """Renaming the folder renames the bot, and so its orders."""
    return config_file.resolve().parent.name


def names_a_file(path: Path) -> bool:
    """A path ending in `.toml` names a configuration file even once that file
    is gone, the shape a path written before a bot was a folder keeps.
    """
    return path.suffix == _TOML_SUFFIX or path.is_file()


def not_a_bot(folder: Path) -> str | None:
    """A registry outlives the folders it names, so a caller skipping a bot
    reads why without a raise.

    Returns:
        Why the path is no bot's folder, None when it holds a `bot.toml`.
    """
    return _refusal(folder, _BOT_FILE_NAME)


def _file_in(folder: Path, file_name: str) -> Path:
    refusal = _refusal(folder, file_name)
    if refusal is not None:
        raise BotCriticalError(refusal)
    return folder / file_name


def _refusal(folder: Path, file_name: str) -> str | None:
    if names_a_file(folder):
        return (
            f"`{folder}` names a file; a bot is a folder holding "
            f"`{_BOT_FILE_NAME}`, so pass its folder, `{folder.parent}`"
        )
    if not (folder / file_name).is_file():
        return f"There is no `{file_name}` in `{folder}`"
    return None
