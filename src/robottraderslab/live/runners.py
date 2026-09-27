import logging

from robottraderslab._core import (
    auto_configure_notebook_logging,
    retry_on_transient,
    run_async,
)
from robottraderslab.bootstrap import (
    AccountConfig,
    BotConfig,
    LiveConfig,
    SecretsByName,
    load_notifiers,
    load_ohlcv_provider,
    load_secrets,
    load_single_account,
    load_strategy,
    require_ohlcv_provider,
)
from robottraderslab.exceptions import ExchangeRecoverableError, StrategyCriticalError
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies import TradingMode, TradingSystem

from .livebot import LiveBot

logger = logging.getLogger(__name__)


def run_livebot(bot_config: BotConfig) -> None:
    """Run one live cycle: read the market, then book and execute on the venue.

    Raises:
        StrategyCriticalError: If the bot declares no `[live]` section.
    """
    auto_configure_notebook_logging()
    run_async(_run_livebot(bot_config))


async def _run_livebot(bot_config: BotConfig) -> None:
    """Run the whole cycle on one event loop so every phase shares its connections."""
    live_config = require_live_config(bot_config)
    config_dir = bot_config.config_dir

    secrets_by_name: SecretsByName = load_secrets(bot_config.secrets_file)
    account = await _load_account(
        live_config.trading_account,
        secrets_by_name,
        max_attempts=live_config.retry.max_attempts,
        base_delay=live_config.retry.base_delay_seconds,
    )
    require_ohlcv_provider(
        live_config.ohlcv_provider, "live.ohlcv_provider", bot_config.config_file
    )
    ohlcv_provider = load_ohlcv_provider(live_config.ohlcv_provider)

    strategy = load_strategy(
        bot_config.strategy.model_dump(),
        account=account,
        trading_system=TradingSystem(trading_mode=TradingMode.LIVE),
        config_dir=config_dir,
    )

    on_fill, on_placement = load_notifiers(live_config.notifier, secrets_by_name)

    livebot = LiveBot(
        strategy,
        ohlcv_provider,
        on_fill=on_fill,
        on_placement=on_placement,
        max_attempts=live_config.retry.max_attempts,
        base_delay=live_config.retry.base_delay_seconds,
    )
    await livebot.run()


async def _load_account(
    account_config: AccountConfig,
    secrets_by_name: SecretsByName,
    *,
    max_attempts: int,
    base_delay: float,
) -> FuturesAccount:
    """No order is out at setup, so a rejection costs nothing to retry, and one
    that outlasts the attempts stops the run before it trades.

    Args:
        secrets_by_name: API credentials keyed by account name.
        max_attempts: Total attempts before the run stops.
        base_delay: Base delay in seconds (doubled on each retry).
    """
    try:
        return await retry_on_transient(
            load_single_account,
            account_config,
            secrets_by_name,
            max_attempts=max_attempts,
            base_delay=base_delay,
            retryable=ExchangeRecoverableError,
        )
    except ExchangeRecoverableError as e:
        logger.error(f"Cannot initialize account: {e}")
        raise


def require_live_config(bot_config: BotConfig) -> LiveConfig:
    """Read the section a run cannot trade without.

    Raises:
        StrategyCriticalError: If the bot declares no `[live]` section.
    """
    if bot_config.live is None:
        raise StrategyCriticalError("`[live]` section is required to trade live")
    return bot_config.live
