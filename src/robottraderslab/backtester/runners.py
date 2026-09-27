from robottraderslab._core import auto_configure_notebook_logging, load_class, run_async
from robottraderslab.bootstrap import (
    BacktestConfig,
    BotConfig,
    load_ohlcv_provider,
    load_simulated_market,
    load_strategy,
    require_ohlcv_provider,
)
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.strategies import TradingMode, TradingSystem

from .backtest_outputs import BacktestOutputs
from .backtester import Backtester

_STRATEGY_ENTRY_POINT_GROUP = "robot_traders_lab.strategies"


def run_backtest(bot_config: BotConfig) -> BacktestOutputs:
    """Replay the configured strategy over history, simulating every fill.

    Returns:
        The run's balances, equity snapshots and fills, ready to analyse.

    Raises:
        StrategyCriticalError: If the bot declares no `[backtest]` section.
    """
    auto_configure_notebook_logging()
    backtest_config = require_backtest_config(bot_config)
    config_dir = bot_config.config_dir

    strategy_class = load_class(
        bot_config.strategy.strategy_class, _STRATEGY_ENTRY_POINT_GROUP
    )

    market = load_simulated_market(
        strategy_class.market_type,
        initial_balance=backtest_config.initial_balance,
        maker_fee_rate=backtest_config.maker_fee_rate,
        taker_fee_rate=backtest_config.taker_fee_rate,
        fee_mode=backtest_config.fee_mode,
        placement_reserve=backtest_config.placement_reserve,
    )

    require_ohlcv_provider(
        backtest_config.ohlcv_provider,
        "backtest.ohlcv_provider",
        bot_config.config_file,
    )
    ohlcv_provider = load_ohlcv_provider(backtest_config.ohlcv_provider)

    strategy = load_strategy(
        bot_config.strategy.model_dump(),
        account=market.account,
        trading_system=TradingSystem(trading_mode=TradingMode.BACKTEST),
        config_dir=config_dir,
    )

    backtester = Backtester(
        strategy,
        market.simulator.simulation_engine,
        ohlcv_provider,
        market.simulator.fill_recorder,
        start_date=backtest_config.start_date,
        end_date=backtest_config.end_date,
        bot_config=bot_config,
        config_dir=config_dir,
    )
    return run_async(backtester.run())


def require_backtest_config(bot_config: BotConfig) -> BacktestConfig:
    """Read the section a backtest cannot run without.

    Raises:
        StrategyCriticalError: If the bot declares no `[backtest]` section.
    """
    if bot_config.backtest is None:
        raise StrategyCriticalError(
            "`[backtest]` section is required to run a backtest"
        )
    return bot_config.backtest
