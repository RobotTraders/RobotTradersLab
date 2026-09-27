import pandas as pd

from .trade_metrics_models import TradeDrawdownMetrics, TradeMetricsResult


class TradeMetricsMixin:
    """The series properties hand out copies, so a reader cannot corrupt the
    measured run.
    """

    _trade_metrics: TradeMetricsResult
    _trade_drawdown: TradeDrawdownMetrics
    _trades: pd.DataFrame

    @property
    def win_rate(self) -> float:
        return self._trade_metrics.win_rate

    @property
    def profit_factor(self) -> float:
        return self._trade_metrics.profit_factor

    @property
    def total_trades(self) -> int:
        return self._trade_metrics.total_trades

    @property
    def winning_trades_count(self) -> int:
        return self._trade_metrics.winning_trades

    @property
    def losing_trades_count(self) -> int:
        return self._trade_metrics.losing_trades

    @property
    def risk_reward_ratio(self) -> float:
        return self._trade_metrics.risk_reward_ratio

    @property
    def max_win_streak(self) -> int:
        return self._trade_metrics.max_win_streak

    @property
    def max_lose_streak(self) -> int:
        return self._trade_metrics.max_lose_streak

    @property
    def total_fees(self) -> float:
        return self._trade_metrics.total_fee

    @property
    def average_fee(self) -> float:
        return self._trade_metrics.avg_fee

    @property
    def biggest_fee(self) -> float:
        return self._trade_metrics.biggest_fee

    @property
    def avg_trade_pnl(self) -> float:
        return self._trade_metrics.avg_trade_pnl

    @property
    def avg_trade_return(self) -> float:
        return self._trade_metrics.avg_trade_return

    @property
    def avg_winning_trade_pnl(self) -> float:
        return self._trade_metrics.avg_winning_trade_pnl

    @property
    def avg_losing_trade_pnl(self) -> float:
        return self._trade_metrics.avg_losing_trade_pnl

    @property
    def largest_winning_trade_pnl(self) -> float:
        return self._trade_metrics.largest_winning_trade_pnl

    @property
    def largest_losing_trade_pnl(self) -> float:
        return self._trade_metrics.largest_losing_trade_pnl

    @property
    def avg_trade_duration_days(self) -> float:
        return self._trade_metrics.avg_trade_duration_days

    @property
    def avg_winning_trade_duration_days(self) -> float:
        return self._trade_metrics.avg_winning_trade_duration_days

    @property
    def avg_losing_trade_duration_days(self) -> float:
        return self._trade_metrics.avg_losing_trade_duration_days

    @property
    def total_pnl(self) -> float:
        return self._trade_metrics.total_pnl

    @property
    def time_in_position_ratio(self) -> float:
        return self._trade_metrics.time_in_position_ratio

    @property
    def avg_trades_per_day(self) -> float:
        return self._trade_metrics.avg_trades_per_day

    @property
    def open_reasons(self) -> pd.Series:
        return self._trade_metrics.open_reasons.copy()

    @property
    def close_reasons(self) -> pd.Series:
        return self._trade_metrics.close_reasons.copy()

    @property
    def max_drawdown_at_trades(self) -> float:
        return self._trade_drawdown.max_drawdown
