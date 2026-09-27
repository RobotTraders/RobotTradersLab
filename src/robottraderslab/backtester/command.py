import logging
from pathlib import Path

from robottraderslab.bootstrap import (
    BotConfig,
    load_log_handlers,
    load_secrets,
    report_unavailable_notifiers,
)
from robottraderslab.exceptions import ExchangeCriticalError, StrategyCriticalError

from .backtest_logging import setup_backtest_logging
from .backtest_outputs import BacktestOutputs
from .runners import require_backtest_config, run_backtest

logger = logging.getLogger(__name__)


def main(
    config: Path,
    logfile_override: Path | None,
    console: bool,
    chart: bool,
    save: bool,
) -> None:
    """Replay the strategy the config declares over historical candles.

    Raises:
        StrategyCriticalError: If the config is unusable.
        ExchangeCriticalError: If the run cannot reach what it needs.
    """
    bot_config = BotConfig.from_file(config)
    config_dir = bot_config.config_dir
    backtest_config = require_backtest_config(bot_config)
    logging_config = backtest_config.logging

    handlers, unavailable = load_log_handlers(
        backtest_config.notifier, load_secrets(bot_config.secrets_file)
    )
    setup_backtest_logging(
        logfile=logfile_override or logging_config.logfile,
        log_name=config.stem,
        console_level=logging_config.console_level,
        file_level=logging_config.file_level,
        enable_console=console,
        enable_file=logging_config.enable_file,
        custom_handlers=handlers,
        config_dir=config_dir,
    )
    report_unavailable_notifiers(unavailable)

    indicators_name = bot_config.strategy.strategy_class
    outputs = run_backtest(bot_config)
    _analyse(outputs, indicators_name=indicators_name, draw_chart=chart, save=save)


def _analyse(
    outputs: BacktestOutputs,
    indicators_name: str,
    draw_chart: bool,
    save: bool,
) -> None:
    analyser = outputs.create_analyser(save=save)
    analyser.print_performance_summary()
    if not draw_chart:
        return
    try:
        analyser.plot_candlesticks(indicators_name=indicators_name)
    except (Exception, ExchangeCriticalError, StrategyCriticalError):
        logger.warning("No chart drawn for `%s`", indicators_name, exc_info=True)
