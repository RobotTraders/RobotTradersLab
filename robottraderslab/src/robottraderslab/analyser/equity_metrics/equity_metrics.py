import numpy as np
import pandas as pd

from .equity_metrics_models import DrawdownMetrics


def calculate_returns(equity_series: pd.Series) -> pd.Series:
    returns = equity_series.pct_change()
    return returns.dropna()


def calculate_roi(equity_series: pd.Series, initial_balance: float) -> float:
    if equity_series.empty:
        return 0.0

    final_capital = float(equity_series.iloc[-1])
    return final_capital / initial_balance - 1


def calculate_hodl_performance(
    initial_capital: float, starting_price: float, ending_price: float
) -> float:
    hodl_capital = initial_capital * (ending_price / starting_price)
    return hodl_capital / initial_capital - 1


def calculate_drawdowns(equity_series: pd.Series) -> DrawdownMetrics:
    equity_ath = equity_series.cummax()
    drawdown = equity_series - equity_ath
    drawdown_pct = drawdown / equity_ath
    max_drawdown = float(drawdown_pct.min())

    return DrawdownMetrics(
        absolute_drawdown=drawdown,
        drawdown_percentage=drawdown_pct,
        max_drawdown=max_drawdown,
        max_drawdown_amount=deepest_drawdown_amount(drawdown, drawdown_pct),
    )


def deepest_drawdown(absolute_drawdown: pd.Series) -> float:
    if absolute_drawdown.empty:
        return 0.0
    return float(absolute_drawdown.min())


def deepest_drawdown_amount(
    absolute_drawdown: pd.Series, drawdown_percentage: pd.Series
) -> float:
    """The amount and the percentage are read from the same moment: as equity
    grows, a later drop can be larger in currency while still being a
    shallower share of the peak it fell from.
    """
    if not drawdown_percentage.notna().any():
        return 0.0
    return float(absolute_drawdown.loc[drawdown_percentage.idxmin()])


def calculate_sharpe_ratio(
    returns: pd.Series,
    annualization_factor: float,
    risk_free_rate: float = 0.0,
) -> float:
    """The risk-free rate is annual and is brought to the period before it is
    netted off.

    Args:
        annualization_factor: Periods in a year (e.g., 365 for daily, 12 for monthly).
        risk_free_rate: Annual risk-free rate as a decimal (e.g., 0.02 for 2%).
    """
    excess_returns = _calculate_excess_returns(
        returns,
        annualization_factor=annualization_factor,
        annual_rate=risk_free_rate,
    )

    mean_excess = float(excess_returns.mean()) if not excess_returns.empty else 0.0
    std_excess = float(excess_returns.std()) if len(excess_returns) > 1 else 0.0
    if std_excess == 0.0:
        return 0.0

    sharpe = mean_excess / std_excess
    if annualization_factor > 0:
        sharpe *= annualization_factor**0.5

    return float(sharpe)


def calculate_sortino_ratio(
    returns: pd.Series,
    annualization_factor: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Only the returns below the risk-free rate count as risk; the rate is
    annual and is brought to the period first.

    Args:
        annualization_factor: Periods in a year (e.g., 365 for daily, 12 for monthly).
        risk_free_rate: Annual risk-free rate as a decimal (e.g., 0.02 for 2%).
    """
    excess_returns = _calculate_excess_returns(
        returns,
        annualization_factor=annualization_factor,
        annual_rate=risk_free_rate,
    )

    downside = excess_returns.copy()
    downside[downside > 0] = 0.0
    downside_squared = downside**2
    mean_downside_squared = (
        float(downside_squared.mean()) if not downside_squared.empty else 0.0
    )
    semideviation = float(mean_downside_squared**0.5)

    if semideviation == 0.0:
        return 0.0

    mean_excess = float(excess_returns.mean()) if not excess_returns.empty else 0.0
    sortino = mean_excess / semideviation
    if annualization_factor > 0:
        sortino *= annualization_factor**0.5

    return float(sortino)


def calculate_calmar_ratio(
    returns: pd.Series, max_drawdown: float, annualization_factor: float
) -> float:
    """Calmar = CAGR / |MaxDrawdown| where CAGR is geometrically annualized.

    Args:
        annualization_factor: Periods per year (e.g., 365 for daily, 12 for monthly).
    """
    if returns.empty or max_drawdown == 0:
        return 0.0

    numeric_returns = returns.astype(float)

    if annualization_factor == 0:
        avg_return = float(numeric_returns.mean())
        return float(avg_return / abs(max_drawdown))
    num_periods = len(numeric_returns)
    arr = numeric_returns.to_numpy(dtype=float, copy=False)
    prod_value = float(np.prod(1.0 + arr))
    total_return = prod_value - 1.0
    cagr = float((1.0 + total_return) ** (annualization_factor / num_periods) - 1.0)
    return float(cagr / abs(max_drawdown))


def calculate_performance_vs_hodl(
    equity_series: pd.Series, price_series: pd.Series, initial_balance: float
) -> tuple[float, float]:
    """The pair reads (hodl_return, performance_vs_hodl), both as decimals."""
    if equity_series.empty or price_series.empty:
        return 0.0, 0.0

    final_capital = float(equity_series.iloc[-1])
    starting_price = float(price_series.iloc[0])
    ending_price = float(price_series.iloc[-1])

    hodl_return = ending_price / starting_price - 1
    hodl_final_capital = initial_balance * (1 + hodl_return)
    performance_vs_hodl = (
        (final_capital - hodl_final_capital) / hodl_final_capital
        if hodl_final_capital != 0
        else 0.0
    )

    return hodl_return, performance_vs_hodl


def _calculate_excess_returns(
    returns: pd.Series,
    *,
    annualization_factor: float,
    annual_rate: float,
) -> pd.Series:
    """
    Args:
        returns: Periodic strategy returns (e.g., daily) as decimals.
        annualization_factor: Periods per year (e.g., 365 for daily, 12 for monthly).
        annual_rate: Annual reference rate (typically risk-free rate or minimum
            acceptable return) as decimal.
    """
    if annualization_factor == 0:
        per_period_rate = annual_rate
    else:
        per_period_rate = float(
            (1.0 + annual_rate) ** (1.0 / annualization_factor) - 1.0
        )

    numeric_returns = returns.astype(float)
    return numeric_returns - per_period_rate
