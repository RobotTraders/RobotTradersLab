import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from robottraderslab._core import (
    Balance,
    Currency,
    Execution,
    OHLCVProviderProtocol,
    PositionSnapshot,
    Symbol,
    assemble_all_ohlcvs_from_requirements,
    retry_on_transient,
    run_async,
)
from robottraderslab.analyser import Analyser
from robottraderslab.bootstrap import (
    BotConfig,
    LiveConfig,
    SecretsByName,
    load_ohlcv_provider,
    load_secrets,
    load_single_account_with_exchange,
    load_strategy,
    require_ohlcv_provider,
)
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.exchanges import FuturesExchangeProtocol
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies import StrategyRequirements, TradingMode, TradingSystem

from ..runners import require_live_config

logger = logging.getLogger(__name__)

_SYMBOL_KEY = "symbol"


def run_report(bot_config: BotConfig, days: int) -> None:
    """Chart the account's performance over the window against the strategy's indicators.

    Candles and indicators are recomputed from the configuration, and the
    account's executions and open positions come from the exchange itself,
    so a report can be produced for any window the venue still remembers
    without the bot logging anything. Candles are fetched from the
    strategy's declared lookback, in published candles, before the window
    opens, so every indicator holds a value from the first reported candle.

    Args:
        bot_config: Loaded configuration, carrying the `[strategy]` and
            `[live]` sections.
        days: How many days back the reported window starts.
    """
    profiles = _profiles(bot_config)
    live_config = require_live_config(bot_config)
    until = datetime.now(UTC)
    since = until - timedelta(days=days)

    symbols = {Symbol.create(profile[_SYMBOL_KEY]) for profile in profiles}
    secrets_by_name: SecretsByName = load_secrets(bot_config.secrets_file)
    require_ohlcv_provider(
        live_config.ohlcv_provider, "live.ohlcv_provider", bot_config.config_file
    )
    ohlcv_provider = load_ohlcv_provider(live_config.ohlcv_provider)
    ohlcv_provider.set_dates(_date_string(since), _date_string(until))

    account_state = run_async(
        _read_the_venue(
            bot_config, live_config, secrets_by_name, since, symbols, ohlcv_provider
        )
    )
    _log_the_read(since, until, account_state)

    analyser = Analyser.from_executions(
        executions=account_state.executions,
        positions=account_state.positions,
        balances=account_state.balances,
        symbols=symbols,
        since=since,
        until=until,
        ohlcv_provider=ohlcv_provider,
        report_config=bot_config.report,
        profiles=profiles,
    )
    analyser.plot_candlesticks(
        indicators_name=bot_config.strategy.strategy_class,
        start_date=_date_string(since),
    )


def _profiles(bot_config: BotConfig) -> list[dict[str, Any]]:
    profiles = bot_config.strategy.profiles
    if not profiles:
        raise StrategyCriticalError("`strategy.profiles` declares nothing to chart")
    return profiles


@dataclass(frozen=True, slots=True)
class _AccountState:
    """What the venue states about the account, read together."""

    executions: list[Execution]
    positions: dict[Symbol, PositionSnapshot]
    balances: dict[Currency, Balance]


def _log_the_read(since: datetime, until: datetime, state: _AccountState) -> None:
    """Name how far back the read actually reached.

    A venue clamps a window to what it retains without saying so, so a short
    answer would otherwise read as a full one.
    """
    covered = f"{_date_string(since)} to {_date_string(until)}"
    if not state.executions:
        logger.info("Covered %s, with no execution in reach", covered)
        return
    oldest = min(execution.timestamp for execution in state.executions)
    logger.info("Covered %s, reaching back to %s", covered, _date_string(oldest))


async def _read_the_venue(
    bot_config: BotConfig,
    live_config: LiveConfig,
    secrets_by_name: SecretsByName,
    since: datetime,
    symbols: set[Symbol],
    ohlcv_provider: OHLCVProviderProtocol,
) -> _AccountState:
    """One event loop serves every read: the venue connection the account
    state comes from is the one the strategy is built on, and the candle
    downloads, bound to the loop that opens them, run inside it too.
    """
    exchange, account = await retry_on_transient(
        load_single_account_with_exchange,
        live_config.trading_account,
        secrets_by_name,
        max_attempts=live_config.retry.max_attempts,
        base_delay=live_config.retry.base_delay_seconds,
    )
    account_state = await _fetch_account_state(exchange, live_config, since, symbols)
    await _warm_ohlcv_provider(bot_config, account, ohlcv_provider)
    return account_state


async def _fetch_account_state(
    exchange: FuturesExchangeProtocol,
    live_config: LiveConfig,
    since: datetime,
    symbols: set[Symbol],
) -> _AccountState:
    """Read the account's executions, open positions and balances in one burst.

    A venue answers a window with what it retains of it, so a position that
    opened before the window and closed inside it arrives as a close with no
    entry.

    The balance is read at the same moment as the executions, so taking the
    window's realised profit back off it lands on the balance the window
    opened with.
    """
    symbols_list = list(symbols)

    def read(fn: Callable[..., Awaitable[Any]], *args: Any) -> Awaitable[Any]:
        return retry_on_transient(
            fn,
            *args,
            max_attempts=live_config.retry.max_attempts,
            base_delay=live_config.retry.base_delay_seconds,
        )

    executions, positions, balances = await asyncio.gather(
        read(exchange.get_executions_since, since, symbols_list),
        read(exchange.get_open_positions, symbols_list),
        read(exchange.get_balances, symbols_list),
    )
    return _AccountState(executions=executions, positions=positions, balances=balances)


async def _warm_ohlcv_provider(
    bot_config: BotConfig,
    account: FuturesAccount,
    ohlcv_provider: OHLCVProviderProtocol,
) -> None:
    """A strategy's `setup` is where its markets and their lookbacks are
    declared, so the report asks it and fetches exactly what it declares.
    """
    strategy = load_strategy(
        bot_config.strategy.model_dump(),
        account=account,
        trading_system=TradingSystem(trading_mode=TradingMode.LIVE),
        config_dir=bot_config.config_dir,
    )
    requirements = StrategyRequirements()
    await strategy.setup(requirements)
    await assemble_all_ohlcvs_from_requirements(ohlcv_provider, requirements.ohlcv)


def _date_string(moment: datetime) -> str:
    return moment.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
