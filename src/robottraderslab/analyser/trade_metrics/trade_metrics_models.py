from dataclasses import dataclass

import pandas as pd

from ..trade_filter import TradeFilter


@dataclass(frozen=True, slots=True)
class TradeCountMetrics:
    """A trade that neither won nor lost still counts toward the total."""

    total_trades: int
    winning_trades: int
    losing_trades: int


@dataclass(frozen=True, slots=True)
class StreakMetrics:
    """A break-even trade ends a winning streak and lengthens a losing one."""

    max_win_streak: int
    max_lose_streak: int


@dataclass(frozen=True, slots=True)
class FeeMetrics:
    """biggest_fee is the single largest fee among those summed into total_fee."""

    total_fee: float
    biggest_fee: float
    avg_fee: float


@dataclass(frozen=True, slots=True)
class AveragePnlMetrics:
    """The return figures are percentages of each trade's entry value, so trades
    of different sizes compare.
    """

    avg_trade_pnl: float
    avg_trade_return: float
    avg_winning_trade_pnl: float
    avg_losing_trade_pnl: float
    avg_winning_trade_return: float
    avg_losing_trade_return: float


@dataclass(frozen=True, slots=True)
class LargestTradeMetrics:
    """The PnL figures are non-negative, even though largest_losing_trade_pnl
    records a loss; the returns keep their sign.
    """

    largest_winning_trade_pnl: float
    largest_losing_trade_pnl: float
    best_trade_return: float
    worst_trade_return: float


@dataclass(frozen=True, slots=True)
class TradeDrawdownMetrics:
    """The deepest fall between two trade completions, in currency and as a share.

    Both are read at the moment the share ran deepest: as equity grows, a
    later fall can cost more currency while costing less of the peak it fell
    from.
    """

    max_drawdown: float
    amount: float | None


@dataclass(frozen=True, slots=True)
class TradeDurationMetrics:
    """Durations run from the entry fill to the exit fill, in fractional days."""

    avg_trade_duration_days: float
    avg_winning_trade_duration_days: float
    avg_losing_trade_duration_days: float


@dataclass(frozen=True, slots=True)
class TimeInPositionMetrics:
    """The share is of the span from the first entry to the last exit, with
    overlapping positions counted once.
    """

    time_in_position_ratio: float
    avg_trades_per_day: float


@dataclass(frozen=True, slots=True)
class ReasonMetrics:
    """The count of trades by the reason each was opened, and by the reason
    each was closed.
    """

    open_reasons: pd.Series
    close_reasons: pd.Series


@dataclass(frozen=True, slots=True)
class TradeMetricsResult:
    """The figures a set of trades yields on its own, whatever subset it is.

    `filter_applied` names the subset, so a figure is never read without
    knowing which trades it was measured over.
    """

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_win_streak: int
    max_lose_streak: int
    total_fee: float
    biggest_fee: float
    avg_fee: float
    avg_trade_pnl: float
    avg_trade_return: float
    avg_winning_trade_pnl: float
    avg_losing_trade_pnl: float
    avg_winning_trade_return: float
    avg_losing_trade_return: float
    largest_winning_trade_pnl: float
    largest_losing_trade_pnl: float
    best_trade_return: float
    worst_trade_return: float
    total_pnl: float
    avg_trade_duration_days: float
    avg_winning_trade_duration_days: float
    avg_losing_trade_duration_days: float
    time_in_position_ratio: float
    avg_trades_per_day: float
    open_reasons: pd.Series
    close_reasons: pd.Series
    risk_reward_ratio: float
    filter_applied: TradeFilter
