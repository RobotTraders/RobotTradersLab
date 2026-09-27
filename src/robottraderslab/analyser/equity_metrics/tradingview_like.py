from typing import Any, cast

import pandas as pd

from robottraderslab._core import OHLCVProviderProtocol, Symbol, last_per_month

from ..equity_sampling import sample_equity_at
from .equity_metrics import deepest_drawdown_amount
from .equity_metrics_models import DrawdownMetrics


def resample_to_monthly(equity_series: pd.Series) -> pd.Series:
    return last_per_month(equity_series).dropna()


def filter_equity_at_trade_times(
    equity_curve: pd.Series, trades: pd.DataFrame
) -> pd.Series:
    """TradingView measures its ratios on the equity at trade entries and exits."""
    moments = pd.concat([trades["entry_time"], trades["exit_time"]])
    return sample_equity_at(equity_curve, moments, include_end=True)


def filter_equity_at_trade_exits(
    equity_curve: pd.Series, trades: pd.DataFrame
) -> pd.Series:
    """TradingView draws its equity and drawdown curves at trade exits alone."""
    return sample_equity_at(equity_curve, trades["exit_time"], include_end=True)


def calculate_tradingview_drawdowns(
    trades: pd.DataFrame,
    initial_capital: float,
    ohlcv_provider: OHLCVProviderProtocol,
) -> DrawdownMetrics:
    """A drawdown is measured against the candle extremes under each open
    trade, so a loss the closes never showed still counts.
    """
    if trades.empty:
        empty_series = pd.Series(dtype=float)
        return DrawdownMetrics(
            absolute_drawdown=empty_series,
            drawdown_percentage=empty_series,
            max_drawdown=0.0,
            max_drawdown_amount=0.0,
        )

    candles_by_symbol = {
        symbol: ohlcv_provider.get_all_cached_ohlcv_for_symbol(Symbol.create(symbol))
        for symbol in trades["symbol"].unique()
    }
    max_equity = initial_capital
    equity_after_previous_trade = initial_capital
    all_drawdowns = []
    all_max_equities = []

    sorted_trades = cast(
        list[dict[str, Any]], trades.sort_values("entry_time").to_dict("records")
    )
    for trade in sorted_trades:
        equity_on_entry = equity_after_previous_trade
        candles = candles_by_symbol[trade["symbol"]].loc[
            trade["entry_time"] : trade["exit_time"]
        ]

        if trade["side"] == "long":
            price_diff = trade["entry_price"] - candles["low"]
        else:
            price_diff = candles["high"] - trade["entry_price"]

        drawdowns = max_equity - equity_on_entry + trade["net_quantity"] * price_diff
        all_drawdowns.append(drawdowns)
        all_max_equities.append(pd.Series(max_equity, index=drawdowns.index))

        equity_after_previous_trade += trade["net_pnl"]
        max_equity = max(max_equity, equity_after_previous_trade)

    combined_drawdowns = pd.concat(all_drawdowns)
    combined_max_equities = pd.concat(all_max_equities)

    drawdown_per_bar = combined_drawdowns.groupby(level=0).max()
    peak_equity_per_bar = combined_max_equities.groupby(level=0).max()

    absolute_drawdown = -drawdown_per_bar.sort_index()
    drawdown_percentage = absolute_drawdown / peak_equity_per_bar
    max_drawdown_pct = float(drawdown_percentage.min())

    return DrawdownMetrics(
        absolute_drawdown=absolute_drawdown,
        drawdown_percentage=drawdown_percentage,
        max_drawdown=max_drawdown_pct,
        max_drawdown_amount=deepest_drawdown_amount(
            absolute_drawdown, drawdown_percentage
        ),
    )


def calculate_tradingview_sortino_ratio(
    returns: pd.Series,
    annualization_factor: float,
    risk_free_rate: float = 0.0,
) -> float:
    """The target return here is the risk-free rate, the traditional finance
    definition of the ratio.

    Args:
        returns: Time series of portfolio returns (typically monthly for TradingView).
        annualization_factor: Periods in a year; TradingView's convention passes zero.
        risk_free_rate: Per-period risk-free rate (e.g., 0.02/12 for 2% annual
            rate monthly).
    """
    per_period_rfr = (
        risk_free_rate
        if annualization_factor == 0
        else float((1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0)
    )
    excess_returns = returns - per_period_rfr

    mean_excess = float(excess_returns.mean()) if not excess_returns.empty else 0.0

    downside = excess_returns.copy()
    downside[downside > 0] = 0.0

    downside_squared_sum = float((downside**2).sum())
    n = len(downside)
    semideviation = float((downside_squared_sum / n) ** 0.5) if n > 0 else 0.0

    if semideviation == 0.0:
        return 0.0
    sortino = mean_excess / semideviation
    if annualization_factor > 0:
        sortino *= annualization_factor**0.5

    return float(sortino)
