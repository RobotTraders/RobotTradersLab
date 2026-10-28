from dataclasses import replace
from datetime import UTC, datetime

from robottraderslab._core import (
    auto_configure_notebook_logging,
    load_class,
    name_process,
    run_async,
)
from robottraderslab.bootstrap import (
    BacktestConfig,
    BotConfig,
    load_ohlcv_provider,
    load_simulated_market,
    load_strategy,
    require_ohlcv_provider,
)
from robottraderslab.exceptions import BotCriticalError
from robottraderslab.strategies import TradingMode, TradingSystem

from .backtest_logging import recorded_log
from .backtest_outputs import BacktestOutputs
from .backtester import Backtester

_STRATEGY_ENTRY_POINT_GROUP = "robot_traders_lab.strategies"


def run_backtest(bot_config: BotConfig, record_log: bool = True) -> BacktestOutputs:
    """Replay the configured strategy over history, simulating every fill.

    Args:
        bot_config: The bot whose `[backtest]` section is replayed.
        record_log: Whether the run's log is held for its run folder; False
            replays without holding it, for a caller that sets up the run's
            logging itself.

    Returns:
        The run's balances, equity snapshots, fills and log, ready to analyse.
        `run_log` holds the records from `[backtest.logging]`'s `file_level`
        up, and is None when `record_log` or its `enable_file` is false.

    Raises:
        BotCriticalError: If the bot declares no `[backtest]` section, or its
            strategy or its OHLCV provider refuses its configuration.
        ExchangeCriticalError: If the venue an exchange-backed
            `[backtest.ohlcv_provider]` reads from is unusable.
    """
    if not record_log:
        return _replay(bot_config)
    auto_configure_notebook_logging()
    logging_config = require_backtest_config(bot_config).logging
    if not logging_config.enable_file:
        return _replay(bot_config)
    with recorded_log(logging_config.file_level) as log_stream:
        outputs = _replay(bot_config)
    return replace(outputs, run_log=log_stream.getvalue())


def _replay(bot_config: BotConfig) -> BacktestOutputs:
    name_process(bot_config.name, datetime.now(UTC))
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
        order_overrides=bot_config.orders.to_overrides(),
    )

    require_ohlcv_provider(
        backtest_config.ohlcv_provider,
        "backtest.ohlcv_provider",
        bot_config.config_file,
    )
    ohlcv_provider = load_ohlcv_provider(backtest_config.ohlcv_provider)

    strategy = load_strategy(
        bot_config.strategy.model_dump(),
        config_file=bot_config.config_file,
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
        BotCriticalError: If the bot declares no `[backtest]` section.
    """
    if bot_config.backtest is None:
        raise BotCriticalError("`[backtest]` section is required to run a backtest")
    return bot_config.backtest
