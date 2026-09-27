import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime

import pandas as pd

from robottraderslab._core import (
    Execution,
    OrderSide,
    Symbol,
    calculate_trade_pnl_pct,
    tag_of,
)

from .analysis_inputs import TRADE_COLUMNS

_QUANTITY_RELATIVE_TOLERANCE = 1e-9
_SIDE_OF = {OrderSide.BUY: "long", OrderSide.SELL: "short"}


@dataclass(frozen=True, slots=True)
class PairedTrades:
    """Closed trades paired from executions, and the closes that yielded none.

    A close is `unpairable` when nothing it could have closed was read, and
    `unpriced` when the venue never stated what it earned. Neither becomes a
    trade: an invented entry, or a profit of zero the venue never stated,
    would put a measurement in the table that nobody made.
    """

    trades: pd.DataFrame
    unpairable: int
    unpriced: int


def build_trades(
    executions: Iterable[Execution],
    *,
    since: datetime,
    until: datetime,
) -> PairedTrades:
    """Pair each symbol's executions into the trades that closed within the window.

    An execution on the side a symbol is already carrying opens or grows the
    position at its own price. One on the opposite side closes that much of
    it, taking the venue's own figure as the trade's gross profit, and its
    fee split in proportion to the quantity each side of the execution
    covers.

    A trade belongs to the window when its closing execution falls inside it.
    An execution outside the window exists only to establish what was already
    open, and never becomes a trade of its own.

    Two kinds of close yield no trade and are counted instead. One the venue
    books a profit on while nothing is open closed a position opened further
    back than the read reached. One the venue states no profit for at all
    earned an amount nobody can recover.

    Args:
        executions: Every execution to read, covering the window and reaching
            back far enough to establish whatever was already open at `since`.
        since: Start of the reported window.
        until: End of the reported window. A trade is kept only when its
            closing execution falls between the two.
    """
    rows: list[dict[str, object]] = []
    unpairable = 0
    unpriced = 0
    for symbol_executions in _by_symbol(executions).values():
        paired, counts = _pair_symbol(
            sorted(symbol_executions, key=lambda execution: execution.timestamp),
            since=since,
            until=until,
        )
        rows.extend(paired)
        unpairable += counts.unpairable
        unpriced += counts.unpriced

    trades = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    if not trades.empty:
        trades = trades.sort_values("entry_time", ascending=False)
    return PairedTrades(trades=trades, unpairable=unpairable, unpriced=unpriced)


@dataclass(frozen=True, slots=True)
class _OpenLeg:
    """The position a symbol is carrying while its executions are being paired."""

    side: str
    quantity: float
    average_price: float
    entry_time: pd.Timestamp
    entry_fee: float
    entry_reason: str | None


def _by_symbol(executions: Iterable[Execution]) -> dict[Symbol, list[Execution]]:
    grouped: dict[Symbol, list[Execution]] = defaultdict(list)
    for execution in executions:
        grouped[execution.symbol].append(execution)
    return grouped


@dataclass(frozen=True, slots=True)
class _UnusableCloses:
    """The closes a symbol's executions yielded no trade for."""

    unpairable: int = 0
    unpriced: int = 0


def _pair_symbol(
    executions: list[Execution],
    *,
    since: datetime,
    until: datetime,
) -> tuple[list[dict[str, object]], _UnusableCloses]:
    leg: _OpenLeg | None = None
    rows: list[dict[str, object]] = []
    unpairable = 0
    unpriced = 0
    for execution in executions:
        inside = since <= execution.timestamp <= until
        profit = execution.realised_profit
        if leg is None:
            if profit:
                if inside:
                    unpairable += 1
                continue
            leg = _opened_by(
                execution, quantity=execution.quantity, fee=execution.fee or 0.0
            )
            continue
        if _SIDE_OF[execution.side] == leg.side:
            leg = _grow(leg, execution)
            continue
        row, leg = _close(leg, execution, profit or 0.0)
        if not inside:
            continue
        if profit is None:
            unpriced += 1
            continue
        rows.append(row)
    return rows, _UnusableCloses(unpairable=unpairable, unpriced=unpriced)


def _grow(leg: _OpenLeg, execution: Execution) -> _OpenLeg:
    """Add to a position already running, at the price the execution paid."""
    quantity = leg.quantity + execution.quantity
    average_price = (
        leg.average_price * leg.quantity + execution.price * execution.quantity
    ) / quantity
    return replace(
        leg,
        quantity=quantity,
        average_price=average_price,
        entry_fee=leg.entry_fee + (execution.fee or 0.0),
    )


def _close(
    leg: _OpenLeg,
    execution: Execution,
    gross_pnl: float,
) -> tuple[dict[str, object], _OpenLeg | None]:
    matched = min(leg.quantity, execution.quantity)
    entry_fee_share = leg.entry_fee * (matched / leg.quantity)
    exit_fee_share = (execution.fee or 0.0) * (matched / execution.quantity)
    net_pnl = gross_pnl - entry_fee_share - exit_fee_share
    row = {
        "entry_time": leg.entry_time,
        "exit_time": pd.Timestamp(execution.timestamp),
        "symbol": str(execution.symbol),
        "side": leg.side,
        "tag": tag_of(execution.client_order_id),
        "entry_price": leg.average_price,
        "exit_price": execution.price,
        "gross_quantity": matched,
        "net_quantity": matched,
        "entry_fee": entry_fee_share,
        "exit_fee": exit_fee_share,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "net_pnl_pct": calculate_trade_pnl_pct(net_pnl, leg.average_price, matched),
        "entry_reason": leg.entry_reason,
        "exit_reason": execution.kind,
    }

    if math.isclose(
        execution.quantity, leg.quantity, rel_tol=_QUANTITY_RELATIVE_TOLERANCE
    ):
        return row, None

    if execution.quantity < leg.quantity:
        remaining = leg.quantity - matched
        return row, replace(
            leg, quantity=remaining, entry_fee=leg.entry_fee - entry_fee_share
        )

    overflow = execution.quantity - matched
    overflow_fee = (execution.fee or 0.0) * (overflow / execution.quantity)
    return row, _opened_by(execution, quantity=overflow, fee=overflow_fee)


def _opened_by(execution: Execution, *, quantity: float, fee: float) -> _OpenLeg:
    return _OpenLeg(
        side=_SIDE_OF[execution.side],
        quantity=quantity,
        average_price=execution.price,
        entry_time=pd.Timestamp(execution.timestamp),
        entry_fee=fee,
        entry_reason=execution.kind,
    )
