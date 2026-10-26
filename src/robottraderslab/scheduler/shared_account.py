import logging
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from robottraderslab._core import bot_leading_bytes, bot_segment
from robottraderslab.bootstrap import SECRET_REFERENCE_KEY
from robottraderslab.exceptions import BotCriticalError

logger = logging.getLogger(__name__)

_NAME_KEY = "name"
_LIVE_KEY = "live"
_TRADING_ACCOUNT_KEY = "trading_account"
_EXCHANGE_KEY = "exchange"
_ACCOUNT_NAME_KEY = "account_name"
_STRATEGY_KEY = "strategy"
_PROFILES_KEY = "profiles"
_SYMBOL_KEY = "symbol"

type _Account = tuple[str, str]
type _Claim = tuple[str, str]

_SEGMENT_KIND = "segment"
_LEADING_BYTES_KIND = "leading bytes"
_SYMBOL_KIND = "symbol"
_REFUSALS = {
    _SEGMENT_KIND: (
        "under the same bot digest `{}`",
        "give each a distinct top-level `name`",
    ),
    _LEADING_BYTES_KIND: (
        "under bot digests sharing their two leading bytes `{}`",
        "give each a distinct top-level `name`",
    ),
    _SYMBOL_KIND: (
        "and both declare the symbol `{}`",
        "give each its own symbols or its own account",
    ),
}


def refuse_shared_account(configs: Iterable[Path]) -> None:
    """A bot's orders are told apart from another bot's by an identity derived
    from its `name`, so two bots sharing that identity on one account read each
    other's orders as their own, and two bots declaring one symbol on one
    account share its position. A configuration that is unreadable, or that
    names no account, is left to its bot's own launch to report.

    Raises:
        BotCriticalError: If two configurations name the same account and
            share a bot identity or a symbol.
    """
    holders: defaultdict[tuple[str, str, str], list[tuple[Path, str]]] = defaultdict(
        list
    )
    for config in configs:
        claims = _claims_of(config)
        if claims is not None:
            (exchange, entry), held = claims
            for kind, value in held:
                holders[entry, kind, value].append((config, exchange))
    for (entry, kind, value), shared in holders.items():
        if len(shared) > 1:
            named = " and ".join(f"`{config}`" for config, _ in shared)
            venue = _venue_of(shared)
            phrase, remedy = _REFUSALS[kind]
            raise BotCriticalError(
                f"{named} trade the account `{entry}`{venue} {phrase.format(value)}; "
                f"{remedy}"
            )


def _claims_of(config: Path) -> tuple[_Account, tuple[_Claim, ...]] | None:
    declaration = _declaration_of(config)
    if declaration is None:
        return None
    name = declaration.get(_NAME_KEY, config.stem)
    if not isinstance(name, str):
        logger.error(
            "The `name` of %s is not a string, comparing nothing for it", config
        )
        return None
    account = _account_of(declaration)
    if account is None:
        logger.debug("%s names no account, comparing nothing for it", config)
        return None
    return account, (
        (_SEGMENT_KIND, bot_segment(name)),
        (_LEADING_BYTES_KIND, bot_leading_bytes(name).hex()),
        *((_SYMBOL_KIND, symbol) for symbol in sorted(_symbols_of(declaration))),
    )


def _declaration_of(config: Path) -> dict[str, Any] | None:
    try:
        return tomllib.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        logger.error(
            "Cannot read %s to compare its bot digest and symbols: %s", config, e
        )
        return None


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


def _venue_of(shared: Iterable[tuple[Path, str]]) -> str:
    declared = {exchange for _, exchange in shared if exchange}
    return f" on {declared.pop()}" if len(declared) == 1 else ""
