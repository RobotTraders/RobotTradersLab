import asyncio
from pathlib import Path

import pandas as pd
import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab._core import PlacementReserve
from robottraderslab.analyser import create_trade_aggregation
from robottraderslab.backtester import BacktestOutputs
from robottraderslab.backtester.backtester import Backtester
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FeeRates,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
    fee_model_for,
)
from robottraderslab.ma_strategy.futures_ma_strategy import FuturesMAStrategy
from robottraderslab.ohlcv_provider import CSVOHLCVProvider
from robottraderslab.strategies import TradingMode, TradingSystem
from robottraderslab.strategies.futures import FuturesAccount

_BTC = Symbol.create("BTC/USDT:USDT")
_TIMEFRAME: TimeFrame = "1d"
_MARGIN_CURRENCY = "USDT"
_INITIAL_BALANCE = 10_000.0
_FEE_RATE = 0.001
_NO_RESERVE = PlacementReserve(margin_markup=0.0, fee_markup=0.0, notional_reserve=0.0)
_START_DATE = "2024-01-01"
_END_DATE = "2024-12-31"

_DIRECTORY = Path(__file__).parent
_PRICES = _DIRECTORY / "data" / "btc_usdt_binance_2024.csv"
_EXPORT = (
    _DIRECTORY / "Test_Lab_simple_btc_BINANCE_BTCUSDT_1d_liste_des_transactions.csv"
)

_EXPORTED_FINAL_EQUITY = 15_581.20
_EXPORTED_COMMISSION = 154.15
_EQUITY_TOLERANCE = 0.20
_COMMISSION_TOLERANCE = 0.01
_PROFIT_TOLERANCE = 0.10


@pytest.fixture(scope="module")
def exchange_mode_outputs() -> BacktestOutputs:
    """TradingView holds nothing back at placement, so every share of the
    reserve stands at zero and the venue modelled here is TradingView itself.
    """
    fill_recorder = CacheFillRecorder()
    simulation_engine = FuturesSimulationEngine(
        initial_balance={_MARGIN_CURRENCY: _INITIAL_BALANCE},
        fee_rates=FeeRates(maker=_FEE_RATE, taker=_FEE_RATE),
        fill_recorder=fill_recorder,
        margin_currency=_MARGIN_CURRENCY,
        fee_model=fee_model_for("exchange", _NO_RESERVE),
    )
    exchange = SimulatedFuturesExchange(simulation_engine, fill_recorder)
    strategy = FuturesMAStrategy(
        account=FuturesAccount(exchange),
        trading_system=TradingSystem(trading_mode=TradingMode.BACKTEST),
        config_dir=Path("bot-config-dir"),
        profiles=[
            {
                "symbol": str(_BTC),
                "timeframe": _TIMEFRAME,
                "available_balance_ratio": 1.0,
                "fast_ma_length": 10,
                "slow_ma_length": 30,
            }
        ],
    )
    backtester = Backtester(
        strategy,
        simulation_engine,
        CSVOHLCVProvider(file=_PRICES, symbol=_BTC, timeframe=_TIMEFRAME),
        fill_recorder,
        _START_DATE,
        _END_DATE,
    )
    return asyncio.run(backtester.run())


@pytest.fixture(scope="module")
def exported_profits() -> list[float]:
    """The export repeats a trade's profit on both of its legs, so one row per
    trade number carries it.
    """
    export = pd.read_csv(_EXPORT, encoding="utf-8")
    profits = (
        export["Profit USDT"]
        .astype(str)
        .str.replace("\u202f", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    export = export.assign(profit=profits).drop_duplicates(subset="Trade #")
    return list(export.sort_values("Trade #")["profit"])


class TestParityWithTheExport:
    """TradingView cuts every entry to Binance's smallest lot and the simulator
    fills to its own step, so the two agree to within what one lot moves on a
    trade.
    """

    def test_final_equity_matches_the_export(self, exchange_mode_outputs):
        final_equity = exchange_mode_outputs.final_balance[_MARGIN_CURRENCY]

        assert abs(final_equity - _EXPORTED_FINAL_EQUITY) < _EQUITY_TOLERANCE

    def test_commission_matches_the_export(self, exchange_mode_outputs):
        commission = exchange_mode_outputs.fills["fee"].sum()

        assert abs(commission - _EXPORTED_COMMISSION) < _COMMISSION_TOLERANCE

    def test_every_trade_profit_matches_the_export(
        self, exchange_mode_outputs, exported_profits
    ):
        trades = create_trade_aggregation(exchange_mode_outputs.fills).trades
        our_profits = list(trades.sort_values("exit_time")["net_pnl"])

        assert len(our_profits) == len(exported_profits)
        for ours, exported in zip(our_profits, exported_profits):
            assert abs(ours - exported) < _PROFIT_TOLERANCE
