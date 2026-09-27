from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import pandas as pd

from robottraderslab._core import Symbol, profile_identity, profile_tag

_SYMBOL_KEY = "symbol"
_TIMEFRAME_KEY = "timeframe"
_TAG_KEY = "tag"
_PROFILE_NAME_COLUMN = "profile_name"


def name_profiles(
    trades: pd.DataFrame, profiles: Sequence[dict[str, Any]]
) -> pd.DataFrame:
    """A market traded by a single profile names every trade on it, whatever the
    orders were tagged with, so a strategy tagging its orders for its own
    purposes is still reported per profile.

    A market several profiles share is told apart by the tag alone, and a
    trade whose tag names none of them keeps no profile at all, since a
    report never states a fact nobody wrote.
    """
    if trades.empty or not profiles:
        return trades
    identities = _identities_by_symbol(profiles)
    named = trades.copy()
    named[_PROFILE_NAME_COLUMN] = [
        _identity_of(str(symbol), tag, identities)
        for symbol, tag in zip(named[_SYMBOL_KEY], named[_TAG_KEY])
    ]
    return named


def _identities_by_symbol(
    profiles: Sequence[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    identities: dict[str, dict[str, str]] = defaultdict(dict)
    for profile in profiles:
        symbol = str(Symbol.create(profile[_SYMBOL_KEY]))
        timeframe = profile[_TIMEFRAME_KEY]
        tag = profile.get(_TAG_KEY) or ""
        identities[symbol][profile_tag(timeframe, tag)] = profile_identity(
            symbol, timeframe, tag
        )
    return identities


def _identity_of(
    symbol: str, tag: object, identities: dict[str, dict[str, str]]
) -> str | None:
    by_tag = identities.get(symbol)
    if by_tag is None:
        return None
    if len(by_tag) == 1:
        return next(iter(by_tag.values()))
    return by_tag.get(str(tag))
