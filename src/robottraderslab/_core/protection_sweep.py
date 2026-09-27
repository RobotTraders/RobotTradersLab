import logging

from .account_requirements import AccountRequirement, AccountRequirements
from .account_snapshot import AccountSnapshot, AccountSnapshots
from .actions import BookKeeper
from .order import OrderType

logger = logging.getLogger(__name__)

_PROTECTION_KINDS = (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT)


def sweep_orphan_protections(
    requirements: AccountRequirements,
    account_snapshots: AccountSnapshots,
    bookkeeper: BookKeeper,
) -> None:
    """Cancel protective orders left resting on a symbol with no position.

    A protective order the venue ties to a position is the venue's to remove
    when that position closes. One placed without that link is nobody's, and
    stays where it is once the position it guarded is gone. Firing with
    nothing to close, it opens a position.

    Args:
        requirements: Account declarations collected in `setup`.
        account_snapshots: The snapshots read for this candle.
        bookkeeper: Collector the cancel actions are added to.
    """
    for requirement in requirements._get_all():
        if not (
            requirement.sweep_protections
            and requirement.positions
            and requirement.open_orders
        ):
            continue
        _sweep_account(
            requirement, account_snapshots.of(requirement.account), bookkeeper
        )


def _sweep_account(
    requirement: AccountRequirement,
    account_snapshot: AccountSnapshot,
    bookkeeper: BookKeeper,
) -> None:
    for symbol in requirement.symbols:
        if account_snapshot.position(symbol) is not None:
            continue
        for order in account_snapshot.open_orders(symbol):
            if order.kind not in _PROTECTION_KINDS:
                continue
            logger.warning(
                f"{symbol}: cancelling a resting {order.kind} order, "
                f"the position it would have closed is gone"
            )
            bookkeeper.add(requirement.account.cancel_order(symbol, order.order_id))
