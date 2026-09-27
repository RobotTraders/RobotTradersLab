from typing import Any

from robottraderslab._core import (
    DEFAULT_FEE_MODE,
    Currency,
    FeeMode,
    OHLCVRow,
    OHLCVsBySymbol,
)

from .base_simulation_engine import BaseSimulationEngine
from .currency_converter import CurrencyConverter, MissingConversionRateError
from .equity_snapshot import EquitySnapshot
from .fees import FeeModel, FeeRates, fee_model_for
from .fill_recorder import CacheFillRecorder, FillRecorder
from .futures_simulated_exchange import SimulatedFuturesExchange
from .futures_simulation_engine import FuturesSimulationEngine
from .order_models import (
    LimitOrder,
    MarketOrder,
    StopLossOrder,
    TakeProfitOrder,
    TriggerOrder,
    reported_reason,
)
from .simulated_position import SimulatedPosition


def create_simulator(
    *,
    initial_balance: dict[Currency, float],
    maker_fee_rate: float,
    taker_fee_rate: float,
    fee_mode: FeeMode = DEFAULT_FEE_MODE,
    **_kwargs: Any,
) -> SimulatedFuturesExchange:
    """Create the futures simulated exchange for backtesting.

    This is the entry point callable for the exchange plugin system; its
    named parameters are the settings the simulator takes.
    """
    return SimulatedFuturesExchange.create_from_settings(
        initial_balance=initial_balance,
        maker_fee_rate=maker_fee_rate,
        taker_fee_rate=taker_fee_rate,
        fee_mode=fee_mode,
    )


__all__ = [
    "BaseSimulationEngine",
    "CacheFillRecorder",
    "CurrencyConverter",
    "EquitySnapshot",
    "FeeModel",
    "FeeRates",
    "FillRecorder",
    "FuturesSimulationEngine",
    "LimitOrder",
    "MarketOrder",
    "MissingConversionRateError",
    "OHLCVRow",
    "OHLCVsBySymbol",
    "SimulatedFuturesExchange",
    "SimulatedPosition",
    "StopLossOrder",
    "TakeProfitOrder",
    "TriggerOrder",
    "create_simulator",
    "fee_model_for",
    "reported_reason",
]
