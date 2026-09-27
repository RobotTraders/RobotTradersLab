import pandas as pd

from robottraderslab._core import FRACTION_TO_PERCENT

from ..equity_sampling import sample_equity_at
from .trade_metrics_models import (
    AveragePnlMetrics,
    FeeMetrics,
    LargestTradeMetrics,
    ReasonMetrics,
    StreakMetrics,
    TimeInPositionMetrics,
    TradeCountMetrics,
    TradeDrawdownMetrics,
    TradeDurationMetrics,
)


def calculate_win_rate(pnl_series: pd.Series) -> float:
    total_trades = len(pnl_series)
    winning_trades = len(pnl_series[pnl_series > 0])
    return float(winning_trades / total_trades if total_trades > 0 else 0)


def calculate_profit_factor(pnl_series: pd.Series) -> float:
    profits = float(pnl_series[pnl_series > 0].sum())
    losses = float(abs(pnl_series[pnl_series < 0].sum()))
    return float(profits / losses if losses != 0 else float("inf"))


def calculate_risk_reward_ratio(avg_metrics: AveragePnlMetrics) -> float:
    win_pct = float(avg_metrics.avg_winning_trade_return)
    lose_pct = float(avg_metrics.avg_losing_trade_return)

    if win_pct == 0.0 and lose_pct == 0.0:
        return 0.0
    if lose_pct == 0.0:
        return float("inf")
    return float(win_pct / abs(lose_pct))


def calculate_streaks(pnl_series: pd.Series) -> StreakMetrics:
    is_win = pnl_series > 0
    streak_changes = is_win.ne(is_win.shift()).cumsum()
    streaks = (
        pd.DataFrame({"win": is_win})
        .groupby(streak_changes)
        .agg({"win": ["sum", "count"]})
    )
    streaks.columns = streaks.columns.map("_".join)

    max_win_streak = streaks.loc[
        streaks["win_sum"] == streaks["win_count"], "win_count"
    ].max()
    max_lose_streak = streaks.loc[streaks["win_sum"] == 0, "win_count"].max()

    return StreakMetrics(
        max_win_streak=max_win_streak if not pd.isna(max_win_streak) else 0,
        max_lose_streak=max_lose_streak if not pd.isna(max_lose_streak) else 0,
    )


def calculate_fee_metrics(fee_series: pd.Series) -> FeeMetrics:
    return FeeMetrics(
        total_fee=float(fee_series.sum()),
        biggest_fee=float(fee_series.max()),
        avg_fee=float(fee_series.mean()),
    )


def calculate_trade_counts(pnl_series: pd.Series) -> TradeCountMetrics:
    if len(pnl_series) == 0:
        return TradeCountMetrics(total_trades=0, winning_trades=0, losing_trades=0)

    winning_trades = len(pnl_series[pnl_series > 0])
    losing_trades = len(pnl_series[pnl_series < 0])

    return TradeCountMetrics(
        total_trades=len(pnl_series),
        winning_trades=winning_trades,
        losing_trades=losing_trades,
    )


def calculate_average_pnl_metrics(
    pnl_series: pd.Series, trades_df: pd.DataFrame
) -> AveragePnlMetrics:
    if len(pnl_series) == 0:
        return AveragePnlMetrics(
            avg_trade_pnl=0.0,
            avg_winning_trade_pnl=0.0,
            avg_losing_trade_pnl=0.0,
            avg_trade_return=0.0,
            avg_winning_trade_return=0.0,
            avg_losing_trade_return=0.0,
        )

    winning_trades = pnl_series[pnl_series > 0]
    losing_trades = pnl_series[pnl_series < 0]

    avg_winning_pnl = float(winning_trades.mean()) if len(winning_trades) > 0 else 0.0
    avg_losing_pnl = float(abs(losing_trades.mean())) if len(losing_trades) > 0 else 0.0

    avg_trade_return = 0.0
    avg_winning_pnl_pct = 0.0
    avg_losing_pnl_pct = 0.0

    if not trades_df.empty:
        winning_trades_df = trades_df[trades_df["net_pnl"] > 0]
        losing_trades_df = trades_df[trades_df["net_pnl"] < 0]

        avg_trade_return = float(trades_df["net_pnl_pct"].mean()) * FRACTION_TO_PERCENT
        avg_winning_pnl_pct = (
            float(winning_trades_df["net_pnl_pct"].mean()) * FRACTION_TO_PERCENT
            if not winning_trades_df.empty
            else 0.0
        )
        avg_losing_pnl_pct = (
            float(losing_trades_df["net_pnl_pct"].mean()) * FRACTION_TO_PERCENT
            if not losing_trades_df.empty
            else 0.0
        )

    return AveragePnlMetrics(
        avg_trade_pnl=float(pnl_series.mean()),
        avg_trade_return=avg_trade_return,
        avg_winning_trade_pnl=avg_winning_pnl,
        avg_losing_trade_pnl=avg_losing_pnl,
        avg_winning_trade_return=avg_winning_pnl_pct,
        avg_losing_trade_return=avg_losing_pnl_pct,
    )


def calculate_largest_trades(
    pnl_series: pd.Series, returns: pd.Series
) -> LargestTradeMetrics:
    """The returns rank the trades by what each made on what it risked, so the
    best and the worst read the same whatever the account held at the time.
    """
    if len(pnl_series) == 0:
        return LargestTradeMetrics(
            largest_winning_trade_pnl=0.0,
            largest_losing_trade_pnl=0.0,
            best_trade_return=0.0,
            worst_trade_return=0.0,
        )

    winning_trades = pnl_series[pnl_series > 0]
    losing_trades = pnl_series[pnl_series < 0]
    return LargestTradeMetrics(
        largest_winning_trade_pnl=(
            float(winning_trades.max()) if len(winning_trades) > 0 else 0.0
        ),
        largest_losing_trade_pnl=(
            float(abs(losing_trades.min())) if len(losing_trades) > 0 else 0.0
        ),
        best_trade_return=float(returns.max()),
        worst_trade_return=float(returns.min()),
    )


def calculate_trade_duration_metrics(trades_df: pd.DataFrame) -> TradeDurationMetrics:
    if trades_df.empty:
        return TradeDurationMetrics(
            avg_trade_duration_days=0.0,
            avg_winning_trade_duration_days=0.0,
            avg_losing_trade_duration_days=0.0,
        )

    winning_trades = trades_df[trades_df["net_pnl"] > 0]
    losing_trades = trades_df[trades_df["net_pnl"] < 0]

    return TradeDurationMetrics(
        avg_trade_duration_days=_calculate_average_duration_days(trades_df),
        avg_winning_trade_duration_days=_calculate_average_duration_days(
            winning_trades
        ),
        avg_losing_trade_duration_days=_calculate_average_duration_days(losing_trades),
    )


def _calculate_average_duration_days(trades_df: pd.DataFrame) -> float:
    if trades_df.empty:
        return 0.0

    duration_seconds = (
        trades_df["exit_time"] - trades_df["entry_time"]
    ).dt.total_seconds()
    duration_days = duration_seconds / (24 * 3600)
    return float(duration_days.mean())


def calculate_time_in_position_metrics(
    trades_df: pd.DataFrame,
) -> TimeInPositionMetrics:
    """Positions open at the same time count their shared time once."""
    if trades_df.empty:
        return TimeInPositionMetrics(
            time_in_position_ratio=0.0,
            avg_trades_per_day=0.0,
        )

    period_start = trades_df["entry_time"].min()
    period_end = trades_df["exit_time"].max()
    trading_period = period_end - period_start

    df = trades_df.loc[:, ["entry_time", "exit_time"]].sort_values("entry_time")
    prev_cummax_end = (
        df["exit_time"].cummax().shift(fill_value=df["entry_time"].iloc[0])
    )
    grp = (df["entry_time"] > prev_cummax_end).cumsum()

    merged = df.groupby(grp, sort=False).agg(
        start=("entry_time", "min"),
        end=("exit_time", "max"),
    )
    total_union_seconds = (merged["end"] - merged["start"]).dt.total_seconds().sum()

    denom = trading_period.total_seconds()
    time_in_position_ratio = float(total_union_seconds / denom) if denom > 0 else 0.0

    trading_days = trading_period.days + 1 if trading_period.days >= 0 else 1
    avg_trades_per_day = len(trades_df) / trading_days if trading_days > 0 else 0.0

    return TimeInPositionMetrics(
        time_in_position_ratio=time_in_position_ratio,
        avg_trades_per_day=avg_trades_per_day,
    )


def calculate_reason_metrics(trades_df: pd.DataFrame) -> ReasonMetrics:
    open_reasons = trades_df["entry_reason"].value_counts()
    open_reasons.name = "entry_reason"
    open_reasons.index.name = None
    close_reasons = trades_df["exit_reason"].value_counts()
    close_reasons.name = "exit_reason"
    close_reasons.index.name = None
    return ReasonMetrics(
        open_reasons=open_reasons,
        close_reasons=close_reasons,
    )


def calculate_trade_based_max_drawdown(
    equity_curve: pd.Series, trades: pd.DataFrame
) -> TradeDrawdownMetrics:
    """Equity is sampled at completions alone, so a fall that recovers before
    the next completion is invisible here. A peak of zero or less states no
    amount for a share to divide by, so the moments it covers are left out of
    the measurement entirely.
    """
    if trades.empty or equity_curve.empty:
        return TradeDrawdownMetrics(max_drawdown=0.0, amount=None)

    equity_at_exits = sample_equity_at(
        equity_curve, trades["exit_time"], include_end=False
    )

    running_max = equity_at_exits.cummax()
    fall = equity_at_exits - running_max
    drawdown = fall / running_max.where(running_max > 0.0)
    if not drawdown.notna().any():
        return TradeDrawdownMetrics(max_drawdown=0.0, amount=None)

    deepest = drawdown.idxmin()
    return TradeDrawdownMetrics(
        max_drawdown=float(drawdown[deepest]), amount=float(fall[deepest])
    )
