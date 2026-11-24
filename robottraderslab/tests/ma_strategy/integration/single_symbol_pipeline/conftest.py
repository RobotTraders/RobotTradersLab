from pathlib import Path

import pytest

from robottraderslab.analyser import Analyser, AnalysisInputs, create_trade_aggregation
from robottraderslab.analyser.trade_aggregator import TradeAggregator
from robottraderslab.backtester import BacktestOutputs, run_backtest
from robottraderslab.bootstrap import BotConfig, ReportConfig


@pytest.fixture(scope="session")
def bot_config() -> BotConfig:
    config_path = Path(__file__).parent / "test_backtest_pipeline.toml"
    return BotConfig.from_file(config_path)


@pytest.fixture(scope="session")
def backtest_outputs(bot_config: BotConfig) -> BacktestOutputs:
    return run_backtest(bot_config)


@pytest.fixture(scope="session")
def trade_aggregator(backtest_outputs: BacktestOutputs) -> TradeAggregator:
    return create_trade_aggregation(backtest_outputs.fills)


@pytest.fixture(scope="session")
def analyser(
    backtest_outputs: BacktestOutputs, trade_aggregator: TradeAggregator
) -> Analyser:
    equity_curve = backtest_outputs.get_equity_curve()
    data = AnalysisInputs(
        trades=trade_aggregator.trades,
        open_positions=trade_aggregator.open_trades,
        equity_curve=equity_curve,
        initial_balance=float(equity_curve.iloc[0]),
    )
    return Analyser.from_analysis_inputs(
        data, backtest_outputs.ohlcv_provider, ReportConfig()
    )
