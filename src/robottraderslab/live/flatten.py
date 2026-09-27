import functools
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from robottraderslab._core import (
    OrderProtocol,
    OrderSide,
    PositionSnapshot,
    StepCount,
    Symbol,
    fetch_account_snapshots,
    retry_on_transient,
    run_async,
    signed_position,
    to_quantity,
    to_step_count,
)
from robottraderslab.bootstrap import (
    AccountConfig,
    BotConfig,
    LiveConfig,
    SecretsByName,
    load_secrets,
    load_single_account_with_exchange,
    load_single_exchange,
    load_strategy,
)
from robottraderslab.exceptions import ExchangeRecoverableError, StrategyCriticalError
from robottraderslab.exchanges import FuturesExchangeProtocol
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies import (
    AccountSnapshots,
    PositionTracker,
    StrategyRequirements,
    TrackingId,
    TradingMode,
    TradingSystem,
)

from .runners import require_live_config

logger = logging.getLogger(__name__)

_VENUE_FILL_RETENTION = timedelta(days=90)
_ONLY_PROFILES = "The account of `%s` holds only what its profiles own"


def main(config: Path, symbol: str | None, external: bool) -> int:
    """Clear the bot's account, of all it carries or of what it never opened.

    Args:
        config: The bot's configuration file, whose `[live.trading_account]`
            names the account and whose secrets file holds its key.
        symbol: Single symbol to flatten; every symbol is flattened when omitted.
        external: Close, on every symbol the bot's profiles declare, the
            quantity no profile of that bot owns, leaving every profile's own
            quantity and every open order where they are.

    Returns:
        1 if the account still carries what the command set out to close;
        0 once the venue is confirmed holding what it should.

    Raises:
        StrategyCriticalError: If the config is unusable, declares no `[live]`
            section, or, for an external run, tracks no profile a position can
            be attributed to.
        ExchangeCriticalError: If the account cannot be loaded.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bot_config = BotConfig.from_file(config)
    live_config = require_live_config(bot_config)
    secrets_by_name = load_secrets(bot_config.secrets_file)
    if external:
        return run_async(
            _flatten_external(bot_config, live_config, secrets_by_name, config.name)
        )
    target_symbol = Symbol.create(symbol) if symbol is not None else None
    return run_async(
        _flatten(
            live_config.trading_account, secrets_by_name, target_symbol, config.name
        )
    )


async def _flatten(
    account_config: AccountConfig,
    secrets_by_name: SecretsByName,
    target_symbol: Symbol | None,
    bot_name: str,
) -> int:
    exchange = await load_single_exchange(account_config, secrets_by_name)
    scope = [target_symbol] if target_symbol is not None else []
    open_orders, open_positions = await _read_open_state(exchange, scope)
    target_symbols = (
        {target_symbol}
        if target_symbol is not None
        else _symbols_with_open_state(open_orders, open_positions)
    )

    if not target_symbols:
        logger.info(
            "The account of `%s` is already flat: nothing to cancel or close", bot_name
        )
        return 0

    for target in sorted(target_symbols):
        await _flatten_symbol(exchange, target, open_positions.get(target))

    return await _verify_flat(exchange, bot_name, target_symbols)


async def _read_open_state(
    exchange: FuturesExchangeProtocol, scope: list[Symbol]
) -> tuple[list[OrderProtocol], dict[Symbol, PositionSnapshot]]:
    open_orders = await retry_on_transient(exchange.get_open_orders, scope)
    open_positions = await retry_on_transient(exchange.get_open_positions, scope)
    return open_orders, open_positions


def _symbols_with_open_state(
    open_orders: list[OrderProtocol], open_positions: dict[Symbol, PositionSnapshot]
) -> set[Symbol]:
    return {order.symbol for order in open_orders} | set(open_positions)


async def _flatten_symbol(
    exchange: FuturesExchangeProtocol, symbol: Symbol, position: PositionSnapshot | None
) -> None:
    try:
        await retry_on_transient(exchange.cancel_orders_for_symbol, symbol)
        logger.info("Cancelled open orders for `%s`", symbol)
        if position is not None:
            await _close_position(exchange, symbol, position)
    except ExchangeRecoverableError as e:
        logger.warning("Could not fully flatten `%s`: %s", symbol, e)


async def _close_position(
    exchange: FuturesExchangeProtocol, symbol: Symbol, position: PositionSnapshot
) -> None:
    close = functools.partial(
        exchange.place_market_order,
        symbol,
        position.side.closing_side,
        position.quantity,
        reduce_only=True,
    )
    await retry_on_transient(close)
    logger.info(
        "Closed %s position on `%s`, quantity %s",
        position.side,
        symbol,
        position.quantity,
    )


async def _verify_flat(
    exchange: FuturesExchangeProtocol, bot_name: str, target_symbols: set[Symbol]
) -> int:
    """A venue whose endpoint takes no symbol answers a scoped read with every
    symbol of the account, so the verdict counts the symbols this run set out
    to flatten and no others.
    """
    open_orders, open_positions = await _read_open_state(exchange, list(target_symbols))
    remaining = _symbols_with_open_state(open_orders, open_positions) & target_symbols
    if remaining:
        logger.error(
            "The account of `%s` is NOT flat: an open order or position remains on %s",
            bot_name,
            sorted(remaining),
        )
        return 1

    logger.info(
        "The account of `%s` is flat on %s: no open order, no open position",
        bot_name,
        ", ".join(str(symbol) for symbol in sorted(target_symbols)),
    )
    return 0


@dataclass(frozen=True, slots=True)
class _Attribution:
    symbol: Symbol
    venue_steps: StepCount
    profiles_steps: StepCount

    @property
    def external_steps(self) -> StepCount:
        return self.venue_steps - self.profiles_steps


async def _flatten_external(
    bot_config: BotConfig,
    live_config: LiveConfig,
    secrets_by_name: SecretsByName,
    bot_name: str,
) -> int:
    exchange, account = await retry_on_transient(
        load_single_account_with_exchange,
        live_config.trading_account,
        secrets_by_name,
        max_attempts=live_config.retry.max_attempts,
        base_delay=live_config.retry.base_delay_seconds,
    )
    attributions = await _attribute_positions(
        bot_config, live_config, account, bot_name
    )
    _report_clean_symbols(attributions)
    external = [
        attribution for attribution in attributions if attribution.external_steps != 0
    ]
    if not external:
        logger.info(_ONLY_PROFILES, bot_name)
        return 0

    for attribution in external:
        if _crosses_flat(attribution):
            _refuse_crossing_flat(attribution)
            continue
        await _close_external_quantity(exchange, attribution, live_config)

    return await _verify_profiles_alone(exchange, live_config, attributions, bot_name)


async def _attribute_positions(
    bot_config: BotConfig,
    live_config: LiveConfig,
    account: FuturesAccount,
    bot_name: str,
) -> list[_Attribution]:
    """Fold the venue's own fills onto the profiles whose tags they carry, so
    what a fold leaves over on a symbol is what no profile of the bot placed. A
    quantity filled before the oldest fill the venue still reports has no
    profile to reach it.

    Raises:
        StrategyCriticalError: If the strategy tracks no profile a position can
            be attributed to.
    """
    requirements = await _declared_requirements(bot_config, account)
    now = datetime.now(UTC)
    account_snapshots = await fetch_account_snapshots(
        requirements.account,
        executions_since=now - _VENUE_FILL_RETENTION,
        max_attempts=live_config.retry.max_attempts,
        base_delay=live_config.retry.base_delay_seconds,
    )
    requirements.tracker.refresh_all(account_snapshots, now, now)
    attributions = _attributions(requirements, account_snapshots)
    if not attributions:
        raise StrategyCriticalError(
            f"The strategy of `{bot_name}` tracks no profile, so nothing on its "
            "account can be told from what it opened; flatten a symbol or pass "
            "--all instead"
        )
    return attributions


async def _declared_requirements(
    bot_config: BotConfig, account: FuturesAccount
) -> StrategyRequirements:
    """A strategy's `setup` is where its profiles name the symbols they trade
    and the tag their own orders carry.
    """
    strategy = load_strategy(
        bot_config.strategy.model_dump(),
        account=account,
        trading_system=TradingSystem(trading_mode=TradingMode.LIVE),
        config_dir=bot_config.config_dir,
    )
    requirements = StrategyRequirements()
    await strategy.setup(requirements)
    return requirements


def _attributions(
    requirements: StrategyRequirements, account_snapshots: AccountSnapshots
) -> list[_Attribution]:
    venue_steps: dict[Symbol, StepCount] = {}
    profiles_steps: dict[Symbol, StepCount] = {}
    for requirement in requirements.tracker._get_all():
        account_snapshot = account_snapshots.of(requirement.account)
        for symbol, tracking_ids in requirement.ids_by_symbol.items():
            venue_steps[symbol] = _position_steps(account_snapshot.position(symbol))
            profiles_steps[symbol] = profiles_steps.get(symbol, 0) + _owned_steps(
                requirement.tracker, tracking_ids
            )
    return [
        _Attribution(
            symbol=symbol,
            venue_steps=venue_steps[symbol],
            profiles_steps=profiles_steps[symbol],
        )
        for symbol in sorted(profiles_steps)
    ]


def _position_steps(position: PositionSnapshot | None) -> StepCount:
    return 0 if position is None else to_step_count(signed_position(position))


def _owned_steps(
    tracker: PositionTracker, tracking_ids: Iterable[TrackingId]
) -> StepCount:
    return sum(
        to_step_count(signed_position(tracked))
        for tracking_id in tracking_ids
        if (tracked := tracker.get(tracking_id)) is not None
    )


def _report_clean_symbols(attributions: Iterable[_Attribution]) -> None:
    for attribution in attributions:
        if attribution.external_steps == 0:
            logger.info(
                "`%s` is clean: the venue holds the %+g its profiles own",
                attribution.symbol,
                to_quantity(attribution.profiles_steps),
            )


async def _close_external_quantity(
    exchange: FuturesExchangeProtocol,
    attribution: _Attribution,
    live_config: LiveConfig,
) -> None:
    quantity = to_quantity(abs(attribution.external_steps))
    side = OrderSide.SELL if attribution.external_steps > 0 else OrderSide.BUY
    close = functools.partial(
        exchange.place_market_order,
        attribution.symbol,
        side,
        quantity,
        reduce_only=_only_reduces(attribution),
    )
    try:
        await retry_on_transient(
            close,
            max_attempts=live_config.retry.max_attempts,
            base_delay=live_config.retry.base_delay_seconds,
        )
    except ExchangeRecoverableError as e:
        logger.warning(
            "Could not close the %+g no profile owns on `%s`: %s",
            to_quantity(attribution.external_steps),
            attribution.symbol,
            e,
        )
        return
    logger.info(
        "`%s`: %s %g removes the %+g no profile owns, leaving the %+g they own",
        attribution.symbol,
        side,
        quantity,
        to_quantity(attribution.external_steps),
        to_quantity(attribution.profiles_steps),
    )


def _crosses_flat(attribution: _Attribution) -> bool:
    """The venue holds the hand's side, so the order restoring the profiles'
    total would carry the position through flat, and a fill no profile placed
    that takes the netted position through flat clears every profile's record
    (`wiki/position-tracking.md`): the profiles would keep their quantity on
    the venue and lose it in their own account of it.
    """
    return attribution.venue_steps * attribution.profiles_steps < 0


def _refuse_crossing_flat(attribution: _Attribution) -> None:
    logger.warning(
        "`%s`: the venue holds %+g against the %+g its profiles own, so removing "
        "the %+g no profile owns would take the position through flat and reset "
        "every profile; flatten the symbol instead and let the bot re-enter",
        attribution.symbol,
        to_quantity(attribution.venue_steps),
        to_quantity(attribution.profiles_steps),
        to_quantity(attribution.external_steps),
    )


def _only_reduces(attribution: _Attribution) -> bool:
    """A venue clamps a reduce-only order at flat."""
    return attribution.profiles_steps * attribution.venue_steps >= 0 and abs(
        attribution.profiles_steps
    ) < abs(attribution.venue_steps)


async def _verify_profiles_alone(
    exchange: FuturesExchangeProtocol,
    live_config: LiveConfig,
    attributions: Sequence[_Attribution],
    bot_name: str,
) -> int:
    """A venue fills what it can of an order and shows the rest only in the
    position it then reports.
    """
    open_positions = await retry_on_transient(
        exchange.get_open_positions,
        [attribution.symbol for attribution in attributions],
        max_attempts=live_config.retry.max_attempts,
        base_delay=live_config.retry.base_delay_seconds,
    )
    unowned = [
        (attribution.symbol, difference)
        for attribution in attributions
        if (
            difference := _position_steps(open_positions.get(attribution.symbol))
            - attribution.profiles_steps
        )
        != 0
    ]
    if unowned:
        logger.error(
            "The account of `%s` still holds what no profile owns: %s",
            bot_name,
            ", ".join(
                f"{symbol} {to_quantity(difference):+g}"
                for symbol, difference in unowned
            ),
        )
        return 1

    logger.info(_ONLY_PROFILES, bot_name)
    return 0
