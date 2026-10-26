import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from .account_requirements import (
    AccountProtocol,
    AccountRequirement,
    AccountRequirements,
)
from .account_snapshot import AccountSnapshot, AccountSnapshots
from .actions import BookKeeper
from .actions.members import carries_this_bot
from .exchange_position_tracker import on_symbol, signed_steps
from .order import Execution, OrderProtocol
from .position import PositionSnapshot
from .position_tracker import position_steps, signed_position
from .quantity import StepCount, to_quantity
from .symbol import Symbol
from .tracked_positions import ATTRIBUTION_REACH

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Holdings:
    position: PositionSnapshot | None
    own_orders: tuple[OrderProtocol, ...]
    other_orders: int


async def clean_removed_profiles(
    requirements: AccountRequirements,
    account_snapshots: AccountSnapshots,
    bookkeeper: BookKeeper,
    *,
    clean: bool,
    max_attempts: int,
    base_delay: float,
) -> None:
    """A profile removed from a bot's configuration leaves its orders and its
    position on the venue with nothing managing them. The position is closed
    reduce-only at market. A position the bot's own fills do not fully explain
    belongs to someone else.

    Args:
        requirements: Account declarations collected in `setup`.
        account_snapshots: The snapshots read for this cycle, whose venue
            reads answer for the whole account.
        clean: Off, the cycle names at WARNING what it would
            have cleaned and leaves it.
        max_attempts: Total retry attempts for transient errors, per call.
        base_delay: Base retry delay in seconds (doubled on each retry).
    """
    for requirement in requirements._get_all():
        if not (requirement.positions and requirement.open_orders):
            continue
        snapshot = account_snapshots.of(requirement.account)
        holdings = _holdings_outside(requirement, snapshot)
        for symbol in sorted(holdings):
            await _clean_symbol(
                requirement.account,
                snapshot,
                symbol,
                holdings[symbol],
                bookkeeper,
                clean=clean,
                max_attempts=max_attempts,
                base_delay=base_delay,
            )


def _holdings_outside(
    requirement: AccountRequirement, snapshot: AccountSnapshot
) -> dict[Symbol, _Holdings]:
    declared = set(requirement.symbols)
    orders: defaultdict[Symbol, list[OrderProtocol]] = defaultdict(list)
    for order in snapshot._account_open_orders():
        if order.symbol not in declared:
            orders[order.symbol].append(order)
    positions = {
        symbol: position
        for symbol, position in snapshot._account_positions().items()
        if symbol not in declared
    }
    return {
        symbol: _holdings_of(positions.get(symbol), orders.get(symbol, []))
        for symbol in positions.keys() | orders.keys()
    }


def _holdings_of(
    position: PositionSnapshot | None, orders: list[OrderProtocol]
) -> _Holdings:
    own = tuple(order for order in orders if carries_this_bot(order.client_order_id))
    return _Holdings(position, own, len(orders) - len(own))


async def _clean_symbol(
    account: AccountProtocol,
    snapshot: AccountSnapshot,
    symbol: Symbol,
    holdings: _Holdings,
    bookkeeper: BookKeeper,
    *,
    clean: bool,
    max_attempts: int,
    base_delay: float,
) -> None:
    window_fills = _own_fills(snapshot._account_executions(), symbol)
    if not (holdings.own_orders or window_fills):
        return
    position = holdings.position
    explained = position is not None and await _explains(
        account, symbol, position, window_fills, max_attempts, base_delay
    )
    closing = position if explained else None
    _name_what_stays(symbol, holdings, closing)
    if closing is None and not holdings.own_orders:
        return
    order_ids = [order.order_id for order in holdings.own_orders]
    _name_what_is_cleaned(symbol, closing, order_ids, clean)
    if clean:
        _book(account, bookkeeper, symbol, closing, order_ids)


def _own_fills(executions: Iterable[Execution], symbol: Symbol) -> list[Execution]:
    return [
        execution
        for execution in on_symbol(executions, symbol)
        if carries_this_bot(execution.client_order_id)
    ]


async def _explains(
    account: AccountProtocol,
    symbol: Symbol,
    position: PositionSnapshot,
    window_fills: list[Execution],
    max_attempts: int,
    base_delay: float,
) -> bool:
    """The candle window reaches back the strategy's longest lookback, which a
    position can outlive.
    """
    held = position_steps(position)
    if _net_steps(window_fills) == held:
        return True
    reach = datetime.now(UTC) - ATTRIBUTION_REACH
    fills = await account._executions_since(
        reach, [symbol], max_attempts=max_attempts, base_delay=base_delay
    )
    return _net_steps(_own_fills(fills, symbol)) == held


def _net_steps(executions: Iterable[Execution]) -> StepCount:
    return sum(signed_steps(execution) for execution in executions)


def _name_what_stays(
    symbol: Symbol, holdings: _Holdings, closing: PositionSnapshot | None
) -> None:
    if holdings.other_orders:
        logger.info(
            f"{symbol}: {holdings.other_orders} open order(s) not placed by this "
            "bot, left in place"
        )
    position = holdings.position
    if position is not None and closing is None:
        logger.warning(
            f"{symbol}: the venue holds {signed_position(position):+g}, which this "
            "bot's own fills do not explain, so the position is left standing"
        )


def _name_what_is_cleaned(
    symbol: Symbol,
    closing: PositionSnapshot | None,
    order_ids: list[str],
    clean: bool,
) -> None:
    template = (
        "closing {} and cancelling"
        if clean
        else "`clean_removed_profiles` is off; it would close {} and cancel"
    )
    logger.warning(
        f"{symbol}: no profile declares it any more, "
        + template.format(_closed_quantity(closing))
        + f" {len(order_ids)} order(s) of this bot: {', '.join(order_ids) or 'none'}"
    )


def _closed_quantity(position: PositionSnapshot | None) -> str:
    if position is None:
        return "no position"
    return f"{to_quantity(position_steps(position)):+g}"


def _book(
    account: AccountProtocol,
    bookkeeper: BookKeeper,
    symbol: Symbol,
    closing: PositionSnapshot | None,
    order_ids: list[str],
) -> None:
    if closing is not None:
        bookkeeper.add(account._close_removed_position(symbol))
    for order_id in order_ids:
        bookkeeper.add(account.cancel_order(symbol, order_id))
