import pandas as pd

from ..trade_filter import TradeFilter
from .trade_metrics import (
    calculate_average_pnl_metrics,
    calculate_fee_metrics,
    calculate_largest_trades,
    calculate_profit_factor,
    calculate_reason_metrics,
    calculate_risk_reward_ratio,
    calculate_streaks,
    calculate_time_in_position_metrics,
    calculate_trade_counts,
    calculate_trade_duration_metrics,
    calculate_win_rate,
)
from .trade_metrics_models import TradeMetricsResult


def compute_trade_metrics(
    trades: pd.DataFrame, trade_filter: TradeFilter = TradeFilter()
) -> TradeMetricsResult:
    """Every figure here holds on any subset of trades, so one measurement serves
    the whole run, one side, one symbol or one profile alike.

    Args:
        trades: Every completed trade of the run.

    Raises:
        ValueError: If an active filter keeps no trade.
    """
    measured = trade_filter.apply_filter(trades)
    if trade_filter.is_active and measured.empty:
        raise ValueError(
            f"No trades found matching filter criteria: {trade_filter.description}. "
            f"Total trades available: {len(trades)}"
        )

    pnl_series = measured["net_pnl"]
    fee_series = measured["entry_fee"] + measured["exit_fee"]

    trade_counts = calculate_trade_counts(pnl_series)
    streaks = calculate_streaks(pnl_series)
    fee_metrics = calculate_fee_metrics(fee_series)
    average_pnl_metrics = calculate_average_pnl_metrics(pnl_series, measured)
    largest_trades = calculate_largest_trades(pnl_series, measured["net_pnl_pct"])
    duration_metrics = calculate_trade_duration_metrics(measured)
    time_position_metrics = calculate_time_in_position_metrics(measured)
    reason_metrics = calculate_reason_metrics(measured)

    return TradeMetricsResult(
        total_trades=trade_counts.total_trades,
        winning_trades=trade_counts.winning_trades,
        losing_trades=trade_counts.losing_trades,
        win_rate=calculate_win_rate(pnl_series),
        profit_factor=calculate_profit_factor(pnl_series),
        max_win_streak=streaks.max_win_streak,
        max_lose_streak=streaks.max_lose_streak,
        total_fee=fee_metrics.total_fee,
        biggest_fee=fee_metrics.biggest_fee,
        avg_fee=fee_metrics.avg_fee,
        avg_trade_pnl=average_pnl_metrics.avg_trade_pnl,
        avg_trade_return=average_pnl_metrics.avg_trade_return,
        avg_winning_trade_pnl=average_pnl_metrics.avg_winning_trade_pnl,
        avg_losing_trade_pnl=average_pnl_metrics.avg_losing_trade_pnl,
        avg_winning_trade_return=average_pnl_metrics.avg_winning_trade_return,
        avg_losing_trade_return=average_pnl_metrics.avg_losing_trade_return,
        largest_winning_trade_pnl=largest_trades.largest_winning_trade_pnl,
        largest_losing_trade_pnl=largest_trades.largest_losing_trade_pnl,
        best_trade_return=largest_trades.best_trade_return,
        worst_trade_return=largest_trades.worst_trade_return,
        total_pnl=float(pnl_series.sum()),
        avg_trade_duration_days=duration_metrics.avg_trade_duration_days,
        avg_winning_trade_duration_days=duration_metrics.avg_winning_trade_duration_days,
        avg_losing_trade_duration_days=duration_metrics.avg_losing_trade_duration_days,
        time_in_position_ratio=time_position_metrics.time_in_position_ratio,
        avg_trades_per_day=time_position_metrics.avg_trades_per_day,
        open_reasons=reason_metrics.open_reasons,
        close_reasons=reason_metrics.close_reasons,
        risk_reward_ratio=calculate_risk_reward_ratio(average_pnl_metrics),
        filter_applied=trade_filter,
    )
