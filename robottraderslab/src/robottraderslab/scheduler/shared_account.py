import logging
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robottraderslab.bootstrap import SECRET_REFERENCE_KEY, bot_name
from robottraderslab.exceptions import BotCriticalError
from robottraderslab.live import working_copy_file
from robottraderslab_core.engine import bot_leading_bytes, bot_segment

from .bot_copy import usable_working_copy

logger = logging.getLogger(__name__)

_LIVE_KEY = "live"
_TRADING_ACCOUNT_KEY = "trading_account"
_EXCHANGE_KEY = "exchange"
_ACCOUNT_NAME_KEY = "account_name"
_STRATEGY_KEY = "strategy"
_PROFILES_KEY = "profiles"
_SYMBOL_KEY = "symbol"

type _Account = tuple[str, str]
type _Key = tuple[str, str, str]

_SEGMENT_KIND = "segment"
_LEADING_BYTES_KIND = "leading bytes"
_SYMBOL_KIND = "symbol"
_UNREADABLE = "Cannot read %s to compare its symbols with the other bots: %s"
_DISTINCT_NAMES = "give each a folder of a distinct name"


def refuse_shared_account(configs: Iterable[Path]) -> dict[Path, str]:
    """A bot's orders are told apart from another bot's by an identity derived
    from its `name`, so two bots sharing that identity on one account read each
    other's orders as their own, and two bots declaring one symbol on one
    account share its position. What a bot runs is compared: its working copy
    when it holds a usable one, its `bot.toml` otherwise. A `bot.toml` that
    differs from its bot's working copy runs nowhere yet, so it is compared only
    on what it adds to its copy, the account, identities and symbols the copy
    does not hold, with what every other bot runs and with what every other
    edited `bot.toml` adds; a conflict there refuses that edit alone. A
    document that cannot be read is logged, and one that names no account
    compares nothing for its bot.

    Args:
        configs: The `bot.toml` path of each registered bot, whether or not
            the file stands.

    Returns:
        For each bot whose edited `bot.toml` conflicts, one line naming the
        edited file, the other bot or edit, the account and the symbol of
        every conflict, which its launch reports; no entry for a bot whose
        edit does not conflict. Each is also logged.

    Raises:
        BotCriticalError: If two bots run documents naming the same account
            and sharing a bot identity or a symbol.
    """
    runs: defaultdict[_Key, list[_Holder]] = defaultdict(list)
    edits: defaultdict[_Key, list[_Holder]] = defaultdict(list)
    for config in configs:
        run_claims, added_claims = _claims_of_bot(config)
        _collect(runs, run_claims)
        _collect(edits, added_claims)
    for key, shared in runs.items():
        if len(shared) > 1:
            raise BotCriticalError(_conflict_of(key, shared, _clearing_of(shared)))
    return _refused_edits(runs, edits)


@dataclass(frozen=True, slots=True)
class _Holder:
    """A bot's claim on an account, and the document it was read from.

    Attributes:
        config: The bot's `bot.toml`, whether or not it stands.
        document: The file the claim was read from, `config` or its working copy.
        exchange: The exchange the document declares, empty when it declares none.
    """

    config: Path
    document: Path
    exchange: str

    @property
    def described(self) -> str:
        if self.document == self.config:
            return f"`{self.config}`"
        return f"the working copy `{self.document}` of `{self.config}`"


def _claims_of_bot(config: Path) -> tuple[dict[_Key, _Holder], dict[_Key, _Holder]]:
    """Read what the bot runs and what its `bot.toml` adds to that."""
    copy = usable_working_copy(config)
    if copy is None:
        return _claims_of_document(config, logging.ERROR), {}
    runs = _claims_in(copy, config, working_copy_file(config), logging.ERROR)
    edited = _claims_of_document(config, logging.DEBUG)
    return runs, {key: holder for key, holder in edited.items() if key not in runs}


def _claims_of_document(config: Path, level: int) -> dict[_Key, _Holder]:
    try:
        document_bytes = config.read_bytes()
    except OSError as e:
        logger.log(level, _UNREADABLE, config, e)
        return {}
    return _claims_in(document_bytes, config, config, level)


def _claims_in(
    document_bytes: bytes, config: Path, document: Path, level: int
) -> dict[_Key, _Holder]:
    try:
        declaration = tomllib.loads(document_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        logger.log(level, _UNREADABLE, document, e)
        return {}
    account = _account_of(declaration)
    if account is None:
        logger.debug("%s names no account, comparing nothing for it", document)
        return {}
    exchange, entry = account
    name = bot_name(config)
    held = (
        (_SEGMENT_KIND, bot_segment(name)),
        (_LEADING_BYTES_KIND, bot_leading_bytes(name).hex()),
        *((_SYMBOL_KIND, symbol) for symbol in sorted(_symbols_of(declaration))),
    )
    return {
        (entry, kind, value): _Holder(config, document, exchange)
        for kind, value in held
    }


def _account_of(declaration: dict[str, Any]) -> _Account | None:
    """The account is named by the credentials entry the account loader draws,
    which this choice must match, since a bot declaring `secret_name` alone
    leaves its exchange to the secrets file.
    """
    live = declaration.get(_LIVE_KEY)
    trading_account = live.get(_TRADING_ACCOUNT_KEY) if isinstance(live, dict) else None
    if not isinstance(trading_account, dict):
        return None
    exchange = _declared_text(trading_account, _EXCHANGE_KEY)
    entry = (
        _declared_text(trading_account, SECRET_REFERENCE_KEY)
        or _declared_text(trading_account, _ACCOUNT_NAME_KEY)
        or exchange
    )
    if not entry:
        return None
    return exchange, entry


def _declared_text(table: dict[str, Any], key: str) -> str:
    value = table.get(key)
    return value if isinstance(value, str) else ""


def _symbols_of(declaration: dict[str, Any]) -> set[str]:
    strategy = declaration.get(_STRATEGY_KEY)
    profiles = strategy.get(_PROFILES_KEY) if isinstance(strategy, dict) else None
    if not isinstance(profiles, list):
        return set()
    return {
        symbol
        for profile in profiles
        if isinstance(profile, dict)
        and (symbol := _declared_text(profile, _SYMBOL_KEY))
    }


def _collect(held: dict[_Key, list[_Holder]], claims: dict[_Key, _Holder]) -> None:
    for key, holder in claims.items():
        held[key].append(holder)


def _conflict_of(key: _Key, holders: list[_Holder], clearing: str) -> str:
    entry, kind, value = key
    named = " and ".join(holder.described for holder in holders)
    where = f"on the account `{entry}`{_venue_of(holders)}"
    if kind == _SYMBOL_KIND:
        return (
            f"{named} trade `{value}` {where}; "
            f"give each bot its own symbols or its own account{clearing}"
        )
    return (
        f"{named} belong to bots {where}, and their folder names share an identity; "
        f"{_DISTINCT_NAMES}{clearing}"
    )


def _venue_of(shared: Iterable[_Holder]) -> str:
    declared = {holder.exchange for holder in shared if holder.exchange}
    return f" on {declared.pop()}" if len(declared) == 1 else ""


def _clearing_of(holders: Iterable[_Holder]) -> str:
    if all(holder.document == holder.config for holder in holders):
        return ""
    return (
        "; after that, delete the working copy or run its bot by hand once "
        "to replace it"
    )


def _refused_edits(
    runs: dict[_Key, list[_Holder]], edits: dict[_Key, list[_Holder]]
) -> dict[Path, str]:
    conflicts: defaultdict[Path, list[str]] = defaultdict(list)
    for key, editing in edits.items():
        for edit in editing:
            running = runs.get(key, [])
            adding = [other for other in editing if other != edit]
            if running or adding:
                conflicts[edit.config].append(
                    _edit_conflict_of(key, edit, running, adding)
                )
    refusals = {
        config: ". ".join(dict.fromkeys(found)) for config, found in conflicts.items()
    }
    for config, refusal in refusals.items():
        logger.debug("The edit of %s is refused: %s", config, refusal)
    return refusals


def _edit_conflict_of(
    key: _Key, edit: _Holder, running: list[_Holder], adding: list[_Holder]
) -> str:
    entry, kind, value = key
    where = f"the account `{entry}`{_venue_of([edit, *running, *adding])}"
    if kind == _SYMBOL_KIND:
        return (
            f"`{edit.config}` adds `{value}` on {where}, "
            f"which {_holding_of(running, adding)}; {_symbol_remedy(running, adding)}"
        )
    others = " or ".join(
        f"`{bot_name(holder.config)}`" for holder in [*running, *adding]
    )
    return (
        f"`{edit.config}` adds {where}, where the folder name of {others} "
        f"shares an identity with its own; {_DISTINCT_NAMES}"
    )


def _holding_of(running: list[_Holder], adding: list[_Holder]) -> str:
    held = []
    if running:
        names = " and ".join(f"`{bot_name(holder.config)}`" for holder in running)
        held.append(f"{names} already {'trades' if len(running) == 1 else 'trade'}")
    if adding:
        files = " and ".join(f"`{holder.config}`" for holder in adding)
        held.append(f"the edited {files} {'adds' if len(adding) == 1 else 'add'} too")
    return " and ".join(held)


def _symbol_remedy(running: list[_Holder], adding: list[_Holder]) -> str:
    which = (
        "one of the two bots"
        if len(running) + len(adding) == 1
        else "all but one of the bots"
    )
    return f"remove it from {which} or give each its own account"
