from dataclasses import dataclass, field

import pandas as pd

from robottraderslab._core import (
    DEFAULT_ANNUALIZATION_FACTOR,
    DEFAULT_CALCULATION_METHOD,
    DEFAULT_RISK_FREE_RATE,
    CalculationMethod,
    OHLCVProviderProtocol,
)

from .equity_metrics import (
    calculate_calmar_ratio,
    calculate_drawdowns,
    calculate_performance_vs_hodl,
    calculate_returns,
    calculate_roi,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    deepest_drawdown,
)
from .equity_metrics_models import EquityMetricsResult, HodlComparison
from .tradingview_like import (
    calculate_tradingview_drawdowns,
    calculate_tradingview_sortino_ratio,
    filter_equity_at_trade_exits,
    filter_equity_at_trade_times,
    resample_to_monthly,
)

_TRADINGVIEW_RISK_FREE_RATE = 0.02 / 12.0
_TRADINGVIEW_ANNUALIZATION_FACTOR = 0.0


def compute_equity_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    ohlcv_provider: OHLCVProviderProtocol,
    price_series: pd.Series | None = None,
    reference_name: str | None = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR,
    calculation_method: CalculationMethod = DEFAULT_CALCULATION_METHOD,
    initial_balance: float | None = None,
) -> EquityMetricsResult:
    """Measure the equity curve the way the calculation method prescribes.

    Args:
        equity_curve: Equity by timestamp.
        trades: Every completed trade of the run.
        ohlcv_provider: Candle source for the intrabar drawdown of the
            TradingView method.
        reference_name: What the price series holds, named on the comparison.
        risk_free_rate: Annual risk-free rate for the Sharpe and Sortino ratios.
        annualization_factor: Periods per year for the daily method.
        initial_balance: Balance the curve is measured from. Every ratio
            that would divide by it is left unstated when it is not given.

    Raises:
        ValueError: If the calculation method is unknown, or if
            "tradingview_like" is requested without an `initial_balance`.
    """
    if calculation_method == "daily":
        return _compute_metrics_daily(
            equity_curve,
            initial_balance,
            price_series,
            reference_name,
            risk_free_rate,
            annualization_factor,
        )
    if calculation_method == "tradingview_like":
        if initial_balance is None:
            raise ValueError(
                "'tradingview_like' measures every ratio against an "
                "initial balance; none was given."
            )
        return _compute_metrics_tradingview_like(
            equity_curve,
            initial_balance,
            trades,
            ohlcv_provider,
            price_series,
            reference_name,
        )
    raise ValueError(
        f"Unknown calculation method '{calculation_method}'. "
        "Valid options are: 'daily', 'tradingview_like'"
    )


def _compute_metrics_daily(
    equity_curve: pd.Series,
    initial_balance: float | None,
    price_series: pd.Series | None,
    reference_name: str | None,
    risk_free_rate: float,
    annualization_factor: float,
) -> EquityMetricsResult:
    daily_equity = _filter_equity_to_midnight(equity_curve)
    drawdown = _stated_drawdown(daily_equity, initial_balance)
    ratios = _ratios_measured_from(
        daily_equity,
        initial_balance,
        drawdown.deepest_share,
        price_series=price_series,
        reference_name=reference_name,
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor,
    )

    return EquityMetricsResult(
        period_start=pd.Timestamp(equity_curve.index[0]),
        period_end=pd.Timestamp(equity_curve.index[-1]),
        initial_equity=initial_balance,
        final_equity=_final_equity(equity_curve, initial_balance),
        net_profit=_net_profit(equity_curve),
        roi=ratios.roi,
        max_drawdown=drawdown.deepest_share,
        max_drawdown_amount=drawdown.deepest_amount,
        sharpe_ratio=ratios.sharpe_ratio,
        sortino_ratio=ratios.sortino_ratio,
        calmar_ratio=ratios.calmar_ratio,
        drawdown_percentage=drawdown.shares,
        absolute_drawdown=drawdown.amounts,
        returns=ratios.returns,
        hodls=ratios.hodls,
        return_over_max_drawdown=ratios.return_over_max_drawdown,
        display_equity_curve=equity_curve,
    )


@dataclass(frozen=True, slots=True)
class _StatedDrawdown:
    """A curve's fall below its own peak, as far as the curve can state it.

    The currency figures hold for any curve. The shares hold only where the
    peak is a balance, and the deepest drop is then read at the moment that
    share was worst, since a later drop on a grown balance can be larger in
    currency while costing less of what was there to lose.
    """

    amounts: pd.Series
    deepest_amount: float
    shares: pd.Series | None
    deepest_share: float | None


def _stated_drawdown(
    equity: pd.Series, initial_balance: float | None
) -> _StatedDrawdown:
    drawdowns = calculate_drawdowns(equity)
    if initial_balance is None:
        return _StatedDrawdown(
            amounts=drawdowns.absolute_drawdown,
            deepest_amount=deepest_drawdown(drawdowns.absolute_drawdown),
            shares=None,
            deepest_share=None,
        )
    return _StatedDrawdown(
        amounts=drawdowns.absolute_drawdown,
        deepest_amount=drawdowns.max_drawdown_amount,
        shares=drawdowns.drawdown_percentage,
        deepest_share=drawdowns.max_drawdown,
    )


@dataclass(frozen=True, slots=True)
class _Ratios:
    """The figures a curve yields only once something states what it opened on.

    A curve stating no balance carries an instance holding nothing, so the
    result is assembled the same way either way.
    """

    roi: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    return_over_max_drawdown: float | None = None
    returns: pd.Series | None = None
    hodls: list[HodlComparison] = field(default_factory=list)


def _ratios_measured_from(
    daily_equity: pd.Series,
    initial_balance: float | None,
    deepest_share: float | None,
    *,
    price_series: pd.Series | None,
    reference_name: str | None,
    risk_free_rate: float,
    annualization_factor: float,
) -> _Ratios:
    """Each of these divides by that balance or by the share of it the deepest
    drawdown represents, so a curve stating neither yields none of them.
    """
    if initial_balance is None or deepest_share is None:
        return _Ratios()

    returns = calculate_returns(daily_equity)
    roi = calculate_roi(daily_equity, initial_balance)
    return _Ratios(
        roi=roi,
        sharpe_ratio=calculate_sharpe_ratio(
            returns,
            annualization_factor=annualization_factor,
            risk_free_rate=risk_free_rate,
        ),
        sortino_ratio=calculate_sortino_ratio(
            returns,
            annualization_factor=annualization_factor,
            risk_free_rate=risk_free_rate,
        ),
        calmar_ratio=calculate_calmar_ratio(
            returns,
            deepest_share,
            annualization_factor=annualization_factor,
        ),
        return_over_max_drawdown=roi / -deepest_share if deepest_share else 0.0,
        returns=returns,
        hodls=_hodl_comparisons(
            daily_equity, price_series, reference_name, initial_balance
        ),
    )


def _net_profit(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    return float(equity_curve.iloc[-1]) - float(equity_curve.iloc[0])


def _final_equity(
    equity_curve: pd.Series, initial_balance: float | None
) -> float | None:
    if initial_balance is None:
        return None
    return float(equity_curve.iloc[-1])


def _hodl_comparisons(
    equity_curve: pd.Series,
    price_series: pd.Series | None,
    reference_name: str | None,
    initial_balance: float,
) -> list[HodlComparison]:
    """The named reference is the only benchmark measured today.

    A run given none is compared against nothing.
    """
    if price_series is None or price_series.empty or reference_name is None:
        return []
    hodl_return, performance_vs_hodl = calculate_performance_vs_hodl(
        equity_curve, price_series, initial_balance
    )
    return [
        HodlComparison(
            held=reference_name,
            hodl_return=hodl_return,
            performance_vs_hodl=performance_vs_hodl,
        )
    ]


def _filter_equity_to_midnight(equity_curve: pd.Series) -> pd.Series:
    """A window too short to contain a midnight has nothing to sample, so the
    curve is measured as it stands, from the points it does have.
    """
    if equity_curve.empty:
        return equity_curve

    dt_index = pd.DatetimeIndex(equity_curve.index)
    midnight_mask = dt_index == dt_index.normalize()
    if not midnight_mask.any():
        return equity_curve
    return equity_curve.loc[midnight_mask]


def _compute_metrics_tradingview_like(
    equity_curve: pd.Series,
    initial_balance: float,
    trades: pd.DataFrame,
    ohlcv_provider: OHLCVProviderProtocol,
    price_series: pd.Series | None,
    reference_name: str | None,
) -> EquityMetricsResult:
    equity_at_trade_times = filter_equity_at_trade_times(equity_curve, trades)

    monthly_equity = resample_to_monthly(equity_at_trade_times)
    if len(monthly_equity) < 2:
        raise ValueError(
            "Insufficient data for monthly sampling, need at least 2 months"
        )
    returns = calculate_returns(monthly_equity)
    roi = calculate_roi(monthly_equity, initial_balance)

    equity_at_exits = filter_equity_at_trade_exits(equity_curve, trades)
    display_drawdown_metrics = calculate_drawdowns(equity_at_exits)

    intrabar_drawdown_metrics = calculate_tradingview_drawdowns(
        trades=trades,
        initial_capital=initial_balance,
        ohlcv_provider=ohlcv_provider,
    )

    sharpe = calculate_sharpe_ratio(
        returns,
        annualization_factor=_TRADINGVIEW_ANNUALIZATION_FACTOR,
        risk_free_rate=_TRADINGVIEW_RISK_FREE_RATE,
    )
    sortino = calculate_tradingview_sortino_ratio(
        returns,
        annualization_factor=_TRADINGVIEW_ANNUALIZATION_FACTOR,
        risk_free_rate=_TRADINGVIEW_RISK_FREE_RATE,
    )
    calmar = calculate_calmar_ratio(
        returns,
        intrabar_drawdown_metrics.max_drawdown,
        annualization_factor=_TRADINGVIEW_ANNUALIZATION_FACTOR,
    )

    return_over_max_drawdown = (
        roi / -intrabar_drawdown_metrics.max_drawdown
        if intrabar_drawdown_metrics.max_drawdown != 0
        else 0.0
    )

    hodls = _hodl_comparisons(
        equity_curve, price_series, reference_name, initial_balance
    )

    return EquityMetricsResult(
        period_start=pd.Timestamp(equity_curve.index[0]),
        period_end=pd.Timestamp(equity_curve.index[-1]),
        initial_equity=initial_balance,
        final_equity=float(equity_curve.iloc[-1]),
        net_profit=_net_profit(equity_curve),
        roi=roi,
        max_drawdown=intrabar_drawdown_metrics.max_drawdown,
        max_drawdown_amount=intrabar_drawdown_metrics.max_drawdown_amount,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        drawdown_percentage=display_drawdown_metrics.drawdown_percentage,
        absolute_drawdown=display_drawdown_metrics.absolute_drawdown,
        returns=returns,
        hodls=hodls,
        return_over_max_drawdown=return_over_max_drawdown,
        display_equity_curve=equity_at_exits,
    )
