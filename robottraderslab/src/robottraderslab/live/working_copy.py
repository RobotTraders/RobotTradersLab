import logging
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robottraderslab._internal import tool_file
from robottraderslab.bootstrap import BotConfig, bot_file, expected_bot_file
from robottraderslab.exceptions import BotCriticalError
from robottraderslab_core import Symbol

logger = logging.getLogger(__name__)

EDIT_REFUSED_VARIABLE = "RTLAB_EDIT_REFUSED"
_WORKING_COPY_SUFFIX = ".working.toml"
_WRITING_SUFFIX = ".writing"
_STRATEGY_KEY = "strategy"
_PROFILES_KEY = "profiles"
_SYMBOL_KEY = "symbol"

type _Document = dict[str, Any]


@dataclass(frozen=True, slots=True)
class TouchedSymbols:
    """The symbols an edit of a bot's configuration touches.

    Attributes:
        symbols: The symbols whose profile sections differ between the two
            files, a symbol only one of them names included.
        every_symbol: Whether the edit touches every symbol of the bot, which
            an edit outside the profile sections does, and so does a file that
            cannot be read or names a symbol that cannot be read.
    """

    symbols: frozenset[Symbol]
    every_symbol: bool

    def covers(self, symbol: Symbol | None) -> bool:
        """Tell whether a failure of the symbol is the edit's.

        A failure of no single symbol (None) always is.
        """
        return self.every_symbol or symbol is None or symbol in self.symbols


@dataclass(frozen=True, slots=True)
class RunningConfig:
    """The configuration a bot's positions were opened under.

    Attributes:
        bot: The validated bot.
        text: The TOML text the bot was built from.
        copy_file: The working copy the bot was built from, None when it was
            built from `bot.toml`.
    """

    bot: BotConfig
    text: str
    copy_file: Path | None


EVERY_SYMBOL = TouchedSymbols(frozenset(), every_symbol=True)
NO_SYMBOL = TouchedSymbols(frozenset(), every_symbol=False)


def working_copy_file(config_file: Path) -> Path:
    """Return where the bot keeps the last configuration that passed verification.

    The file sits in the `.rtlab` folder beside the configuration file.
    """
    return tool_file(config_file, _WORKING_COPY_SUFFIX)


def load_bot_file(
    folder: Path, content_file: Path | None = None
) -> tuple[BotConfig, bytes]:
    """Load the bot declared by its `bot.toml`, or by the file given in its place.

    The bytes are loaded as the folder's `bot.toml`, so the bot is the folder's.

    Args:
        folder: The bot's folder, holding its `bot.toml`.
        content_file: The file to read in place of `bot.toml`, None to read
            `bot.toml`.

    Returns:
        The validated bot and the bytes it was loaded from.

    Raises:
        BotCriticalError: If the path names a file, the folder holds no
            `bot.toml` or the bytes declare no bot the engine can run.
        OSError: If the file read cannot be.
        TypeError: If a path the bytes declare, such as `secrets_file`, is not
            a string.
    """
    config = bot_file(folder)
    content = (content_file or config).read_bytes()
    return BotConfig.from_bytes(content, config), content


def load_running_config(bot_folder: Path) -> RunningConfig:
    """Load the configuration the bot's positions were opened under.

    That is the bot's working copy when it holds one it can use, and its
    `bot.toml` otherwise. A copy that cannot be read or is not valid TOML counts
    as none and logs a WARNING naming the file. The copy is loaded as if it were
    `bot.toml`, so it names the same bot, and `bot.toml` is not read, so it may
    be missing or not valid TOML. Nothing is written, and no strategy code runs.

    Args:
        bot_folder: The bot's folder, holding its `bot.toml` unless the bot
            holds a usable working copy.

    Returns:
        The validated bot, whose `config_file` is the folder's `bot.toml`, with
        the TOML text of the file it was loaded from and the working copy when
        that is the file.

    Raises:
        BotCriticalError: If the bot holds no usable working copy and the path
            names a file, the folder holds no `bot.toml` or `bot.toml` declares
            no bot the engine can run; or if the working copy declares no bot
            the engine can run, naming the copy.
        OSError: If `bot.toml` is read and cannot be.
        TypeError: If a path the loaded file declares, such as `secrets_file`,
            is not a string.
    """
    config = expected_bot_file(bot_folder)
    copy_file = working_copy_file(config)
    working_copy = read_working_copy(copy_file)
    if working_copy is None:
        bot, content = load_bot_file(bot_folder)
        return RunningConfig(bot, content.decode(), None)
    bot = bot_of_working_copy(working_copy, config, copy_file)
    return RunningConfig(bot, working_copy.decode(), copy_file)


@contextmanager
def naming_the_copy(running: RunningConfig) -> Iterator[None]:
    """Name the working copy in a `BotCriticalError` the block raises.

    The error is re-raised as it is when the bot was built from `bot.toml`.
    """
    try:
        yield
    except BotCriticalError as error:
        if running.copy_file is None:
            raise
        raise working_copy_error(running.copy_file, error) from error


def bot_of_working_copy(
    working_copy: bytes, config: Path, copy_file: Path
) -> BotConfig:
    """Build the bot a working copy declares, as if it were the `bot.toml` at `config`.

    Raises:
        BotCriticalError: If the copy declares no bot the engine can run,
            naming the copy at `copy_file`.
    """
    try:
        return BotConfig.from_bytes(working_copy, config)
    except BotCriticalError as error:
        raise working_copy_error(copy_file, error) from error


def read_working_copy(path: Path, level: int = logging.WARNING) -> bytes | None:
    """Read the working copy, None when the bot holds none it can use.

    A copy that cannot be read or is not valid TOML counts as none and logs a
    message naming the file at `level`, a WARNING unless the caller reads the
    copy of a bot it does not run and reports an unusable copy itself or
    leaves that to the bot's own cycle.
    """
    if not path.exists():
        return None
    try:
        content = path.read_bytes()
    except OSError as e:
        logger.log(level, "Working copy `%s` cannot be read, ignored: %s", path, e)
        return None
    if _parsed(content) is None:
        logger.log(level, "Working copy `%s` is not valid TOML, ignored", path)
        return None
    return content


def working_copy_error(path: Path, error: BotCriticalError) -> BotCriticalError:
    """Return the error met while the working copy is the file read, naming that copy.

    The copy is loaded as a `bot.toml`, so the message says so, since the
    error's own text may name that file.
    """
    return BotCriticalError(f"Working copy `{path}`, loaded as `bot.toml`: {error}")


def write_working_copy(path: Path, content: bytes) -> bool:
    """Replace the working copy with the bytes, creating its folder when missing.

    A reader meets the previous copy or the new one whole, never part of one.
    A write that fails logs an ERROR naming the file and leaves the previous
    copy in place.

    Returns:
        Whether the copy now holds the bytes.
    """
    writing = path.with_name(path.name + _WRITING_SUFFIX)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        writing.write_bytes(content)
        writing.replace(path)
    except OSError as e:
        logger.error("Cannot write the working copy `%s`: %s", path, e)
        return False
    return True


def touched_symbols(working_copy: bytes, edited: bytes | None) -> TouchedSymbols:
    """Compare an edited configuration with the working copy, as parsed values.

    A symbol is touched when its profile sections, taken in order, differ
    between the two. Every symbol is touched when anything outside the
    profile sections differs, when the edited file is missing (None) or is not
    valid TOML, or when a section of either names a symbol that cannot be read.

    Args:
        working_copy: The working copy's bytes, valid TOML.
        edited: The configuration file's bytes, None when it could not be read.
    """
    copy_document = tomllib.loads(working_copy.decode())
    edited_document = _parsed(edited) if edited is not None else None
    if edited_document is None:
        return EVERY_SYMBOL
    if _without_profiles(copy_document) != _without_profiles(edited_document):
        return EVERY_SYMBOL
    copy_sections = _sections_by_symbol(copy_document)
    edited_sections = _sections_by_symbol(edited_document)
    if copy_sections is None or edited_sections is None:
        return EVERY_SYMBOL
    return TouchedSymbols(
        frozenset(
            symbol
            for symbol in copy_sections.keys() | edited_sections.keys()
            if copy_sections.get(symbol) != edited_sections.get(symbol)
        ),
        every_symbol=False,
    )


def _parsed(content: bytes) -> _Document | None:
    try:
        return tomllib.loads(content.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        logger.debug("Not valid TOML: %s", e)
        return None


def _without_profiles(document: _Document) -> _Document:
    strategy = document.get(_STRATEGY_KEY)
    if not isinstance(strategy, dict):
        return document
    return {
        **document,
        _STRATEGY_KEY: {
            key: value for key, value in strategy.items() if key != _PROFILES_KEY
        },
    }


def _sections_by_symbol(document: _Document) -> dict[Symbol, list[Any]] | None:
    """Group the profile sections by symbol, in order.

    None when a section has no readable symbol.
    """
    strategy = document.get(_STRATEGY_KEY)
    sections = strategy.get(_PROFILES_KEY, []) if isinstance(strategy, dict) else []
    if not isinstance(sections, list):
        return None
    grouped: dict[Symbol, list[Any]] = {}
    for section in sections:
        symbol = _symbol_of(section)
        if symbol is None:
            return None
        grouped.setdefault(symbol, []).append(section)
    return grouped


def _symbol_of(section: Any) -> Symbol | None:
    if not isinstance(section, dict) or not isinstance(section.get(_SYMBOL_KEY), str):
        return None
    try:
        return Symbol.create(section[_SYMBOL_KEY])
    except ValueError as e:
        logger.debug("Section symbol not readable: %s", e)
        return None
