import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import partial
from itertools import chain
from pathlib import Path

from robottraderslab._internal import (
    EX_TEMPFAIL,
    DryRunFailure,
    OrderOverrides,
    auto_configure_notebook_logging,
    collecting_refusals,
    failure_of,
    hold_instance_lock,
)
from robottraderslab.bootstrap import (
    AccountConfig,
    BotConfig,
    LiveConfig,
    SecretsByName,
    bot_name,
    load_notifiers,
    load_ohlcv_provider,
    load_secrets,
    load_single_account,
    load_strategy,
    require_ohlcv_provider,
)
from robottraderslab.exceptions import (
    BotCriticalError,
    ExchangeCriticalError,
    ExchangeRecoverableError,
)
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies import (
    StrategyProtocol,
    StrategyRequirements,
    TradingMode,
    TradingSystem,
)
from robottraderslab_core import Symbol
from robottraderslab_core.engine import (
    OnOrderFilled,
    OnOrderPlaced,
    name_process,
    run_async,
)
from robottraderslab_core.exchanges import OHLCVProviderProtocol, retry_on_transient

from .livebot import LiveBot
from .working_copy import NO_SYMBOL, TouchedSymbols

logger = logging.getLogger(__name__)


def run_alone(config: Path, run: Callable[[], int]) -> int:
    """Run the config's cycle unless another process is running it.

    One instance of a config acts on its accounts at a time, since two would
    act twice.

    Returns:
        The run's exit code, or `EX_TEMPFAIL` when another process holds the
        config.
    """
    with hold_instance_lock(config) as held:
        if not held:
            logger.warning(
                "%s is already trading in another process, exiting",
                bot_name(config),
            )
            return EX_TEMPFAIL
        return run()


def run_livebot(bot_config: BotConfig) -> None:
    """Run one live cycle: read the market, then book and execute on the venue.

    Raises:
        BotCriticalError: If the bot declares no `[live]` section.
        ExchangeRecoverableError: If the trading account cannot be loaded after
            its retries.
    """
    run_livebot_at(bot_config, datetime.now(UTC), NO_SYMBOL)


def run_livebot_at(
    bot_config: BotConfig, moment: datetime, exit_only: TouchedSymbols
) -> None:
    """Run one live cycle at the moment, the touched symbols in exit-only mode.

    The cycle books the candles that closed at the moment, however late it
    starts. A bot touched on every symbol runs every symbol its strategy's
    `setup` declares in exit-only mode.

    Raises:
        BotCriticalError: If the bot declares no `[live]` section, or cannot run.
        ExchangeCriticalError: If the venue cannot be traded on.
        ExchangeRecoverableError: If the trading account cannot be loaded after
            its retries.
    """
    auto_configure_notebook_logging()
    name_process(bot_config.name, moment)
    run_async(_run_livebot(bot_config, moment, exit_only))


def dry_run_livebot(bot_config: BotConfig, moment: datetime) -> list[DryRunFailure]:
    """Run one live cycle at the moment with nothing sent, and return what failed.

    The run loads the account, builds the strategy and runs the cycle as a
    dry run, logging nothing above DEBUG, the build's own lines included.
    Whatever stops the load, the build or the cycle is one failure naming no
    symbol, a `BotCriticalError` and an `ExchangeCriticalError` included, and
    so is a bot declaring no `[live]` section. An `ExchangeCriticalError` met
    by the cycle follows the failures the cycle had met before it, and a stop
    before the cycle follows the refusals of the profile sections the build had
    met.

    Returns:
        Every failure the run met, in the order it met them; empty when the
        whole cycle ran through.
    """
    return contained_dry_run(lambda: run_async(_dry_run_livebot(bot_config, moment)))


def contained_dry_run(
    dry_run_steps: Callable[[], list[DryRunFailure]],
) -> list[DryRunFailure]:
    """Run the steps of a dry run, turning an error they raise into one failure.

    An `Exception`, a `BotCriticalError` or an `ExchangeCriticalError` they raise
    is one failure naming no symbol, which follows the refusals of the profile
    sections the build had met when the error stopped the steps.

    Args:
        dry_run_steps: The steps to run, returning the failures they met.

    Returns:
        What the steps returned, or the failures of what stopped them.
    """
    with collecting_refusals() as refusals:
        try:
            return dry_run_steps()
        except (Exception, BotCriticalError, ExchangeCriticalError) as error:
            logger.debug("A dry run stopped on %r", error, exc_info=True)
            return [*refusals, failure_of(error, None)]


def require_live_config(bot_config: BotConfig) -> LiveConfig:
    """Read the section a run cannot trade without.

    Raises:
        BotCriticalError: If the bot declares no `[live]` section.
    """
    if bot_config.live is None:
        raise BotCriticalError("`[live]` section is required to trade live")
    return bot_config.live


async def _run_livebot(
    bot_config: BotConfig, moment: datetime, exit_only: TouchedSymbols
) -> None:
    """Run the whole cycle on one event loop so every phase shares its connections."""
    live_config = require_live_config(bot_config)
    secrets_by_name: SecretsByName = load_secrets(bot_config.secrets_file)
    strategy, ohlcv_provider = await _build_strategy_and_provider(
        bot_config, live_config, secrets_by_name
    )
    on_fill, on_placement = load_notifiers(live_config.notifier, secrets_by_name)
    exit_only_symbols = (
        await _declared_symbols(strategy)
        if exit_only.every_symbol
        else exit_only.symbols
    )
    livebot = _livebot(
        strategy,
        ohlcv_provider,
        live_config,
        moment,
        on_fill=on_fill,
        on_placement=on_placement,
        exit_only_symbols=exit_only_symbols,
    )
    await livebot.run()


async def _dry_run_livebot(
    bot_config: BotConfig, moment: datetime
) -> list[DryRunFailure]:
    live_config = require_live_config(bot_config)
    with _debug_logs_only():
        strategy, ohlcv_provider = await _build_strategy_and_provider(
            bot_config, live_config, load_secrets(bot_config.secrets_file)
        )
        livebot = _livebot(strategy, ohlcv_provider, live_config, moment, dry_run=True)
        try:
            await livebot.run()
        except ExchangeCriticalError as error:
            logger.debug("A dry run stopped on %r", error, exc_info=True)
            return [*livebot.failures, failure_of(error, None)]
    return livebot.failures


async def _build_strategy_and_provider(
    bot_config: BotConfig, live_config: LiveConfig, secrets_by_name: SecretsByName
) -> tuple[StrategyProtocol, OHLCVProviderProtocol]:
    """Load the account and the candle source, then build the strategy on them."""
    account = await _load_account(
        live_config.trading_account,
        secrets_by_name,
        bot_config.orders.to_overrides(),
        max_attempts=live_config.retry.max_attempts,
        base_delay=live_config.retry.base_delay_seconds,
    )
    require_ohlcv_provider(
        live_config.ohlcv_provider, "live.ohlcv_provider", bot_config.config_file
    )
    ohlcv_provider = load_ohlcv_provider(live_config.ohlcv_provider)
    strategy = load_strategy(
        bot_config.strategy.model_dump(),
        config_file=bot_config.config_file,
        account=account,
        trading_system=TradingSystem(trading_mode=TradingMode.LIVE),
        config_dir=bot_config.config_dir,
    )
    return strategy, ohlcv_provider


async def _load_account(
    account_config: AccountConfig,
    secrets_by_name: SecretsByName,
    order_overrides: OrderOverrides,
    *,
    max_attempts: int,
    base_delay: float,
) -> FuturesAccount:
    """Load the trading account, retrying a rejection.

    No order is out at setup, so a rejection costs nothing to retry, and one
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
            order_overrides,
            max_attempts=max_attempts,
            base_delay=base_delay,
            retryable=ExchangeRecoverableError,
        )
    except ExchangeRecoverableError as e:
        logger.error(f"Cannot initialize account: {e}")
        raise


async def _declared_symbols(strategy: StrategyProtocol) -> frozenset[Symbol]:
    """The symbols the strategy's `setup` declares on its accounts."""
    requirements = StrategyRequirements()
    await strategy.setup(requirements)
    return frozenset(
        chain.from_iterable(
            requirement.symbols for requirement in requirements.account._get_all()
        )
    )


def _livebot(
    strategy: StrategyProtocol,
    ohlcv_provider: OHLCVProviderProtocol,
    live_config: LiveConfig,
    moment: datetime,
    *,
    on_fill: Iterable[OnOrderFilled] = (),
    on_placement: Iterable[OnOrderPlaced] = (),
    exit_only_symbols: frozenset[Symbol] = frozenset(),
    dry_run: bool = False,
) -> LiveBot:
    return LiveBot(
        strategy,
        ohlcv_provider,
        on_fill=on_fill,
        on_placement=on_placement,
        clean_removed_profiles=live_config.clean_removed_profiles,
        max_attempts=live_config.retry.max_attempts,
        base_delay=live_config.retry.base_delay_seconds,
        exit_only_symbols=exit_only_symbols,
        dry_run=dry_run,
        clock=partial(_at, moment),
    )


def _at(moment: datetime) -> datetime:
    return moment


@contextmanager
def _debug_logs_only() -> Iterator[None]:
    """Keep every line above DEBUG from the log's handlers within the block."""
    handlers = list(logging.getLogger().handlers)
    for handler in handlers:
        handler.addFilter(_at_debug)
    try:
        yield
    finally:
        for handler in handlers:
            handler.removeFilter(_at_debug)


def _at_debug(record: logging.LogRecord) -> bool:
    return record.levelno <= logging.DEBUG
