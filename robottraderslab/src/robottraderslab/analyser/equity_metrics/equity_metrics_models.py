from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class DrawdownMetrics:
    """The share is meaningful only where the peak is a balance. A curve tracking
    what trading earned peaks on an amount earned, so its shares are computed
    the same way and left unstated by whoever reads them.
    """

    absolute_drawdown: pd.Series
    drawdown_percentage: pd.Series
    max_drawdown: float
    max_drawdown_amount: float


@dataclass(frozen=True, slots=True)
class HodlComparison:
    """How the run compares against holding something instead.

    `held` names what the comparison holds, so a report can say which
    benchmark its figures are measured against.
    """

    held: str
    hodl_return: float
    performance_vs_hodl: float


@dataclass(frozen=True, slots=True)
class EquityMetricsResult:
    """A figure measured against the balance the curve opened on is absent when
    the run states no balance. Those are the ones typed as optional here, and
    `hodls`, which is then empty. The figures read straight off the curve are
    stated either way, and are the ones typed as required.
    """

    period_start: pd.Timestamp
    period_end: pd.Timestamp
    initial_equity: float | None
    final_equity: float | None
    net_profit: float
    roi: float | None
    max_drawdown: float | None
    max_drawdown_amount: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    calmar_ratio: float | None
    drawdown_percentage: pd.Series | None
    absolute_drawdown: pd.Series
    returns: pd.Series | None
    hodls: list[HodlComparison]
    return_over_max_drawdown: float | None
    display_equity_curve: pd.Series
