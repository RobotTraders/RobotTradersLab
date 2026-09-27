from collections.abc import Iterable
from datetime import datetime, timedelta

import pandas as pd

from robottraderslab._core import Execution


def build_equity_curve(
    executions: Iterable[Execution],
    *,
    since: datetime,
    until: datetime,
    opening_balance: float | None,
) -> pd.Series:
    """Accumulate what trading booked across the window, from the balance it opened on.

    Each execution states the profit and fee it booked at its own moment, so
    the curve adds that profit less that fee as the moment passes. Every point
    is then a balance the account held, which is what makes a share of the
    curve's own peak a share of an account.

    The curve carries a point at every moment something was booked, which
    keeps a rise and the fall after it both visible however short the window.
    It also carries every UTC midnight, because the daily figures are sampled
    there, and the window's own two edges, so the curve reaches exactly as
    far as the candles it is drawn against.
    """
    booked = sorted(executions, key=lambda execution: execution.timestamp)
    checkpoints = sorted(
        {
            since,
            *_midnights(since, until),
            *_moments_booked(booked, since, until),
            until,
        }
    )
    opened_on = opening_balance if opening_balance is not None else 0.0
    values = _running_totals(checkpoints, booked, opened_on)
    return pd.Series(values, index=pd.DatetimeIndex(checkpoints))


def recover_opening_balance(
    current_balance: float, executions: Iterable[Execution]
) -> float | None:
    """State what the account held when the window opened, from what it holds now.

    Realised profit carries no deposit and no withdrawal, so taking it back
    off the current balance measures what trading did to an account of that
    size.

    Returns:
        The balance the window opened on, or None where the arithmetic leaves
        nothing positive for a share of it to divide by.
    """
    booked = sum(_booked_amount(execution) for execution in executions)
    opening_balance = current_balance - booked
    return opening_balance if opening_balance > 0.0 else None


def _moments_booked(
    booked: list[Execution], since: datetime, until: datetime
) -> list[datetime]:
    return [
        execution.timestamp
        for execution in booked
        if since <= execution.timestamp <= until
    ]


def _midnights(since: datetime, until: datetime) -> list[datetime]:
    first = _next_midnight(since)
    if first > until:
        return []
    return pd.date_range(start=first, end=until, freq="D").to_pydatetime().tolist()


def _next_midnight(moment: datetime) -> datetime:
    start_of_day = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_of_day if moment == start_of_day else start_of_day + timedelta(days=1)


def _running_totals(
    checkpoints: list[datetime], booked: list[Execution], opening_balance: float
) -> list[float]:
    cursor = 0
    running = opening_balance
    totals = []
    for checkpoint in checkpoints:
        while cursor < len(booked) and booked[cursor].timestamp <= checkpoint:
            running += _booked_amount(booked[cursor])
            cursor += 1
        totals.append(running)
    return totals


def _booked_amount(execution: Execution) -> float:
    return (execution.realised_profit or 0.0) - (execution.fee or 0.0)
