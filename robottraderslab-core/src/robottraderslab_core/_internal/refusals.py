import logging
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .exceptions import (
    ExchangeRecoverableError,
    InsufficientMarginError,
    NoOpenPositionError,
)
from .symbol import Symbol

logger = logging.getLogger(__name__)

_MAX_PROFILES_NAMED = 10
_MAX_ANSWERS_QUOTED = 3
_NO_PROFILE = "no profile, the order carries no tag"

_collected: ContextVar[list[tuple[str, str]] | None] = ContextVar(
    "margin_refusals", default=None
)


def report_refusal(
    symbol: Symbol,
    tag: str | None,
    error: ExchangeRecoverableError,
    *,
    is_exit: bool,
    what: str,
) -> None:
    """Report a refusal the venue gave an order at the level it owes.

    An exit refused is an ERROR, since the strategy's position is not closed,
    unless the venue answers that no position is held, which is an INFO for
    any order, since nothing is left to close. An entry refused for lack of
    margin joins the line of its cycle when its call reports margin refusals
    together, and any other refusal is a WARNING.

    Args:
        symbol: The market of the order.
        tag: The tag the order's client order id carries, which names the
            profile; None for an order that carries none.
        error: The venue's answer.
        is_exit: Whether the order closes exposure the strategy holds.
        what: How the order is named in the line.
    """
    refused = _collected.get()
    if isinstance(error, NoOpenPositionError):
        logger.info(
            "%s not placed, the account holds no position on %s (%s): %s",
            what,
            symbol,
            _whose(symbol, tag),
            error,
        )
    elif is_exit:
        logger.error(
            "Exit refused, the strategy's position is not closed: %s, %s: %s",
            subject_of(symbol, tag),
            what,
            error,
        )
    elif isinstance(error, InsufficientMarginError) and refused is not None:
        refused.append((_line_name(symbol, tag), str(error)))
    else:
        logger.warning("Exchange rejected %s (%s): %s", what, symbol, error)


@contextmanager
def collecting_margin_refusals() -> Iterator[None]:
    """Hold the margin refusals of the block, then report them in one line.

    Inside the block an entry the venue refuses for lack of margin is held,
    and the block's end writes one WARNING for all of them.
    """
    refused: list[tuple[str, str]] = []
    token = _collected.set(refused)
    try:
        yield
    finally:
        _collected.reset(token)
        if refused:
            logger.warning(_fully_committed(refused))


def subject_of(symbol: Symbol, tag: str | None) -> str:
    """Name the market and the profile of an exit, as the line of its refusal does."""
    return f"{symbol}, {_whose(symbol, tag)}"


def _whose(symbol: Symbol, tag: str | None) -> str:
    """Name the profile an order belongs to, as its client order id tells."""
    if not tag:
        return _NO_PROFILE
    return f"profile {_line_name(symbol, tag)}"


def _line_name(symbol: Symbol, tag: str | None) -> str:
    return f"{symbol}@{tag}" if tag else str(symbol)


def _fully_committed(refused: list[tuple[str, str]]) -> str:
    counts = Counter(name for name, _ in refused)
    named = [
        name if count == 1 else f"{name} x{count}"
        for name, count in list(counts.items())[:_MAX_PROFILES_NAMED]
    ]
    if len(counts) > _MAX_PROFILES_NAMED:
        named.append(f"and {len(counts) - _MAX_PROFILES_NAMED} more")
    answers = list(dict.fromkeys(answer for _, answer in refused))
    return (
        "The account is fully committed, its free margin carries no more orders: "
        f"{len(refused)} not placed ({', '.join(named)}), the venue answered: "
        f"{'; '.join(answers[:_MAX_ANSWERS_QUOTED])}"
    )
