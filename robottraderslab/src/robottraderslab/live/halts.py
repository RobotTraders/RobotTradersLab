import logging
from collections.abc import Collection, Iterable, Mapping, Sequence

from robottraderslab._internal import (
    carried_orders,
    non_empty_by_account,
    rebuilt_actions,
)
from robottraderslab.futures import FuturesOrderBatchAction, FuturesOrderModifyAction
from robottraderslab.strategies import AccountSnapshots
from robottraderslab_core import Symbol
from robottraderslab_core.engine import AccountProtocol, ActionID, Halt
from robottraderslab_core.exchanges import BaseExchangeAction, TradingStatus
from robottraderslab_core.strategies import AccountRequirements

from .exit_only import restrict_account

logger = logging.getLogger(__name__)


def log_halts(halts: Iterable[Halt], level: int) -> None:
    """Log once each halted symbol, at the logging level given.

    The line names the symbol, the venue, the account and the venue's status
    word.

    Args:
        level: A level of the `logging` module, such as `logging.WARNING`.
    """
    for halt in halts:
        logger.log(
            level,
            "%s on %s (account %s) is %s: the venue reports status %s",
            halt.symbol,
            halt.venue,
            halt.account,
            halt.closure.status,
            halt.closure.venue_status,
        )


def held_back(
    requirements: AccountRequirements,
    halts: Collection[Halt],
    status: TradingStatus,
) -> dict[AccountProtocol, frozenset[Symbol]]:
    """Return the symbols each account's venue reports at the status.

    A halt holds back the account whose listing reports it and no other.

    Args:
        requirements: Account declarations collected in `setup`.
        halts: The halts of the cycle.
        status: The status the symbols are halted at.
    """
    return non_empty_by_account(
        (
            requirement.account,
            frozenset(
                halt.symbol
                for halt in halts
                if halt.account == requirement.account.name
                and halt.closure.status is status
            ),
        )
        for requirement in requirements._get_all()
    )


def closed_with_unlisted(
    closed: Mapping[AccountProtocol, frozenset[Symbol]],
    unlisted_by_account: Mapping[AccountProtocol, frozenset[Symbol]],
) -> dict[AccountProtocol, frozenset[Symbol]]:
    """Join the symbols a listing lacks to those closed to every order.

    A symbol its venue does not list is sent nothing for the cycle, as one the
    venue closed to every order is. An account keeps the symbols closed on it
    and gains the unlisted ones.

    Args:
        closed: The symbols closed to every order, by account.
        unlisted_by_account: The declared symbols the account's venue does not
            list, by account.
    """
    joined = dict(closed)
    for account, symbols in unlisted_by_account.items():
        joined[account] = joined.get(account, frozenset()) | symbols
    return joined


def restrict_halted(
    requirements: AccountRequirements,
    account_snapshots: AccountSnapshots,
    held: Mapping[AccountProtocol, frozenset[Symbol]],
    actions: Sequence[BaseExchangeAction],
) -> list[BaseExchangeAction]:
    """Run each held account's symbols in exit-only mode.

    The actions that go to the account's venue are judged against its own
    positions and open orders, and every other action stays as booked.

    Args:
        requirements: Account declarations collected in `setup`.
        account_snapshots: The snapshots read for this candle, which hold the
            positions and open orders of the held accounts.
        held: The symbols closed to new positions, by account.
        actions: Everything the strategy booked this candle, in booking order.

    Returns:
        The cancels the mode books, then the actions let through, in booking
        order.
    """
    cancels: list[BaseExchangeAction] = []
    for requirement in requirements._get_all():
        symbols = held.get(requirement.account)
        if not symbols:
            continue
        mine = [action for action in actions if requirement.account._books(action)]
        allowed = restrict_account(
            requirement, account_snapshots, symbols, mine, cancels
        )
        mine_ids = {action.id for action in mine}
        rebuilt = {action.id: action for action in allowed}
        actions = [
            rebuilt[action.id] if action.id in mine_ids else action
            for action in actions
            if action.id not in mine_ids or action.id in rebuilt
        ]
    return [*cancels, *actions]


def without_actions_on(
    closed: Mapping[AccountProtocol, Collection[Symbol]],
    actions: Sequence[BaseExchangeAction],
) -> list[BaseExchangeAction]:
    """Leave out every action on a symbol closed on the account it goes to.

    A member of a batch or of a modification on one of them goes too, and the
    rest keeps its booking order.

    Args:
        closed: The symbols closed to every order, by account.
        actions: Everything booked this candle, in booking order.
    """
    return rebuilt_actions(actions, _ids_on(closed, actions), {})


def _ids_on(
    closed: Mapping[AccountProtocol, Collection[Symbol]],
    actions: Sequence[BaseExchangeAction],
) -> set[ActionID]:
    gone: set[ActionID] = set()
    for action in actions:
        symbols = _closed_where_it_goes(closed, action)
        if isinstance(action, (FuturesOrderBatchAction, FuturesOrderModifyAction)):
            gone.update(
                order.id for order in carried_orders(action) if order.symbol in symbols
            )
        elif action.symbol in symbols:
            gone.add(action.id)
    return gone


def _closed_where_it_goes(
    closed: Mapping[AccountProtocol, Collection[Symbol]], action: BaseExchangeAction
) -> Collection[Symbol]:
    for account, symbols in closed.items():
        if account._books(action):
            return symbols
    return ()
