import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from decimal import Decimal

from robottraderslab_core import Symbol
from robottraderslab_core.engine import AccountProtocol, AccountRequirement
from robottraderslab_core.exchanges import PositionSnapshot
from robottraderslab_core.strategies import AccountRequirements, AccountSnapshots

logger = logging.getLogger(__name__)


def report_unlisted_holdings(
    declarations: AccountRequirements,
    account_snapshots: AccountSnapshots,
    level: int,
) -> None:
    """Report what an account still holds on the symbols its venue does not list.

    A listing lacking a symbol says nothing of what the account holds on it,
    so the position and the orders the snapshot shows are named, once per
    cycle.

    Args:
        declarations: Declarations asking each account for its positions and
            its open orders on its unlisted symbols alone, which the snapshots
            hold the reads of. An account that read neither is left out.
        account_snapshots: The snapshots of the cycle.
        level: A level of the `logging` module, such as `logging.ERROR`.
    """
    for account_requirement in declarations._get_all():
        positions, order_counts = _holdings_among(
            account_requirement, account_snapshots
        )
        for symbol in sorted(positions.keys() | order_counts.keys()):
            _log_holding(symbol, positions.get(symbol), order_counts[symbol], level)


async def listing_lacks(
    declarations: AccountRequirements,
) -> dict[AccountProtocol, frozenset[Symbol]]:
    """Return the declared symbols each account's venue does not list.

    Args:
        declarations: The strategy's declarations of the accounts.

    Returns:
        The unlisted symbols by account, none for an account with none of them.
    """
    return non_empty_by_account(
        [
            (
                account_requirement.account,
                await account_requirement.account._unlisted_among(
                    account_requirement.symbols
                ),
            )
            for account_requirement in declarations._get_all()
        ]
    )


async def listing_lacks_among(
    declarations: AccountRequirements, symbols: Iterable[Symbol]
) -> dict[AccountProtocol, frozenset[Symbol]]:
    """Return the symbols among those given that an account declares and lacks.

    Args:
        declarations: The strategy's declarations of the accounts.
        symbols: The symbols to look for.

    Returns:
        The unlisted symbols by account, none for an account with none of them.
    """
    asked = frozenset(symbols)
    return non_empty_by_account(
        [
            (
                account_requirement.account,
                await account_requirement.account._unlisted_among(
                    asked.intersection(account_requirement.symbols)
                ),
            )
            for account_requirement in declarations._get_all()
        ]
    )


def non_empty_by_account(
    pairs: Iterable[tuple[AccountProtocol, frozenset[Symbol]]],
) -> dict[AccountProtocol, frozenset[Symbol]]:
    """Return the symbols of each account, leaving out an account with none.

    Args:
        pairs: Each account with its symbols.
    """
    return {account: symbols for account, symbols in pairs if symbols}


def split_by_read(
    declarations: AccountRequirements,
    unlisted_by_account: Mapping[AccountProtocol, frozenset[Symbol]],
) -> tuple[
    dict[AccountProtocol, frozenset[Symbol]], dict[AccountProtocol, frozenset[Symbol]]
]:
    """Split the unlisted symbols by whether their account read its holdings.

    Args:
        declarations: The declarations the cycle read the accounts' state with.
        unlisted_by_account: The unlisted symbols by account.

    Returns:
        The unlisted symbols of the accounts that read their positions, their
        open orders and their trigger entries, and those of the accounts that
        did not.
    """
    reading = {
        account_requirement.account
        for account_requirement in declarations._get_all()
        if account_requirement.positions
        and account_requirement.open_orders
        and account_requirement.triggers
    }
    read = non_empty_by_account(
        (account, symbols)
        for account, symbols in unlisted_by_account.items()
        if account in reading
    )
    unread = non_empty_by_account(
        (account, symbols)
        for account, symbols in unlisted_by_account.items()
        if account not in reading
    )
    return read, unread


def _holdings_among(
    account_requirement: AccountRequirement, account_snapshots: AccountSnapshots
) -> tuple[dict[Symbol, PositionSnapshot], Counter[Symbol]]:
    if not (account_requirement.positions or account_requirement.open_orders):
        return {}, Counter()
    account_snapshot = account_snapshots.of(account_requirement.account)
    declared = frozenset(account_requirement.symbols)
    positions = (
        account_snapshot._account_positions() if account_requirement.positions else {}
    )
    counts = (
        Counter(order.symbol for order in account_snapshot._account_open_orders())
        if account_requirement.open_orders
        else Counter()
    )
    return (
        {symbol: positions[symbol] for symbol in positions.keys() & declared},
        Counter({symbol: counts[symbol] for symbol in counts.keys() & declared}),
    )


def _log_holding(
    symbol: Symbol, position: PositionSnapshot | None, order_count: int, level: int
) -> None:
    if position is not None:
        logger.log(
            level,
            "%s: the venue lists no contract for it, yet the account holds "
            "%s %s; its position is not settled and nothing is sent on it",
            symbol,
            position.side,
            _plain(position.quantity),
        )
    if order_count:
        logger.log(
            level,
            "%s: the venue lists no contract for it, yet %d order(s) rest on it; "
            "nothing is sent on it",
            symbol,
            order_count,
        )


def _plain(quantity: float) -> str:
    return format(Decimal(repr(quantity)).normalize(), "f")
