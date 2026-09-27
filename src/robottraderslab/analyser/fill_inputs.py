import pandas as pd

from robottraderslab._core import OHLCVProviderProtocol
from robottraderslab.bootstrap import ReportConfig

from .analysis_inputs import AnalysisInputs
from .reference_price_handler import resolve_reference
from .trade_aggregator import create_trade_aggregation


def build_fill_inputs(
    fills: pd.DataFrame,
    equity_curve: pd.Series,
    ohlcv_provider: OHLCVProviderProtocol,
    *,
    report_config: ReportConfig,
) -> AnalysisInputs:
    """The curve is expected to open on the balance the run started with:
    its first point is what every ratio divides by.
    """
    aggregation = create_trade_aggregation(fills)
    reference_price, reference_symbol_str = resolve_reference(
        report_config.reference_symbol,
        report_config.reference_timeframe,
        equity_curve.index,
        ohlcv_provider,
    )
    return AnalysisInputs(
        trades=aggregation.trades,
        open_positions=aggregation.open_trades,
        equity_curve=equity_curve,
        initial_balance=float(equity_curve.iloc[0]),
        reference_price=reference_price,
        reference_symbol_str=reference_symbol_str,
    )
