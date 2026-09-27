import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime

from robottraderslab._core import (
    BASE_DELAY_SECONDS,
    MAX_ATTEMPTS,
    RESTING_ENTRY_KINDS,
    VENUE_FIRED_SOURCES,
    AccountRequirements,
    BaseExchangeAction,
    CandleWindows,
    Execution,
    FillDescriber,
    FillEffect,
    FillSource,
    Notifications,
    OHLCVProviderProtocol,
    OnOrderFilled,
    OnOrderPlaced,
    OrderFill,
    OrderType,
    TimeFrame,
    assemble_available_ohlcvs_from_requirements,
    candle_closes_at,
    candle_windows,
    confirm_booked_protections,
    execute_trading_actions,
    fetch_account_snapshots,
    hold_margin_targets,
    log_strategy_slots,
    retry_on_transient,
    sweep_orphan_protections,
    widen_tracked_positions,
)
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
    StrategyCriticalError,
)
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    OHLCVs,
    StrategyProtocol,
    StrategyRequirements,
)

logger = logging.getLogger(__name__)


class LiveBot:
    """Multi-timeframe live trading bot for executing trading strategies."""

    def __init__(
        self,
        strategy: StrategyProtocol,
        ohlcv_provider: OHLCVProviderProtocol,
        *,
        on_fill: Iterable[OnOrderFilled] = (),
        on_placement: Iterable[OnOrderPlaced] = (),
        max_attempts: int = MAX_ATTEMPTS,
        base_delay: float = BASE_DELAY_SECONDS,
        requirements_factory: Callable[[], StrategyRequirements] = StrategyRequirements,
    ):
        """Initialise a multi-timeframe live trading bot.

        Args:
            on_fill: Notified once for each fill of a cycle, whatever told
                the engine about it. Subscribing one is also what puts a
                stop-loss, take-profit or liquidation the venue fired within
                reach, on every account trading a declared symbol.
            on_placement: Notified once for each order the venue accepted.
            max_attempts: Total retry attempts for transient errors.
            base_delay: Base delay in seconds (doubled on each retry).
            requirements_factory: Builds the requirements `setup` declares
                into, fresh for each cycle.
        """
        self._strategy = strategy
        self._ohlcv_provider = ohlcv_provider
        self._requirements_factory = requirements_factory
        self._on_fill = list(on_fill)
        self._report_fills = bool(self._on_fill)
        self._on_placement = list(on_placement)
        self._max_attempts = max_attempts
        self._base_delay = base_delay

    async def run(self) -> None:
        """Run one complete trading cycle."""
        self._log_cycle_header()
        await self._run_cycle()

    async def _run_cycle(self) -> None:
        requirements = await retry_on_transient(
            self._load_strategy,
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
        )
        self._log_trading_accounts(requirements.account)
        log_strategy_slots(self._strategy)

        declared_timeframes = {r.timeframe for r in requirements.ohlcv._get_all()}
        now = datetime.now(UTC)
        closing_timeframes = {
            timeframe
            for timeframe in declared_timeframes
            if candle_closes_at(timeframe, now)
        }
        if declared_timeframes and not closing_timeframes:
            logger.info("No declared candle closes this minute, nothing to do")
            return

        ohlcvs = await self._fetch_ohlcvs(requirements)
        if ohlcvs is None:
            logger.error("No declared market data could be fetched, skipping cycle")
            return

        closing_timeframes = {
            timeframe
            for timeframe in closing_timeframes
            if ohlcvs._has_timeframe(timeframe)
        }
        if not closing_timeframes:
            return

        try:
            self._strategy.generate_trading_signals(ohlcvs)
        except Exception as e:
            logger.error("Signal generation failed, skipping cycle: %s", e)
            return

        try:
            account_snapshots = await fetch_account_snapshots(
                requirements.account,
                executions_since=ohlcvs._earliest_candle_open(),
                report_fills=self._report_fills,
                max_attempts=self._max_attempts,
                base_delay=self._base_delay,
            )
        except ExchangeRecoverableError as e:
            logger.warning("Account state fetch failed, skipping cycle: %s", e)
            return
        await hold_margin_targets(
            requirements.account,
            account_snapshots,
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
        )

        closed = candle_windows(requirements.ohlcv, ohlcvs, closing_timeframes)
        reported_since = min(window.start for window in closed.values())
        reported_until = max(window.end for window in closed.values())
        try:
            if requirements.tracker.refresh_all(
                account_snapshots, reported_since, reported_until
            ):
                await widen_tracked_positions(
                    requirements,
                    account_snapshots,
                    executions_since=ohlcvs._earliest_candle_open(),
                    reported_since=reported_since,
                    reported_until=reported_until,
                    max_attempts=self._max_attempts,
                    base_delay=self._base_delay,
                )
        except ExchangeRecoverableError as e:
            logger.warning("Tracked position settlement failed, skipping cycle: %s", e)
            return
        except Exception as e:
            logger.error(f"Internal error refreshing tracked positions: {e}")
            return

        notifications = Notifications(self._on_fill, self._on_placement)
        executed = await self._book_timeframes(
            requirements, ohlcvs, account_snapshots, closing_timeframes, notifications
        )
        await confirm_booked_protections(
            executed, max_attempts=self._max_attempts, base_delay=self._base_delay
        )
        await self._notify_entry_fills(
            requirements, ohlcvs, account_snapshots, closing_timeframes, notifications
        )
        await self._notify_venue_fired_fills(
            requirements, ohlcvs, account_snapshots, closing_timeframes, notifications
        )
        await notifications.flush()

    async def _fetch_ohlcvs(self, requirements: StrategyRequirements) -> OHLCVs | None:
        return await assemble_available_ohlcvs_from_requirements(
            self._ohlcv_provider,
            requirements.ohlcv,
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
        )

    async def _load_strategy(self) -> StrategyRequirements:
        """Load strategy with fresh requirements to avoid duplicates on retry."""
        requirements = self._requirements_factory()
        await self._strategy.setup(requirements)
        return requirements

    def _log_cycle_header(self) -> None:  # pragma: no cover
        logger.info("=" * 80)
        logger.info("LiveBot Trading Cycle Started")

    def _log_trading_accounts(self, accounts: AccountRequirements) -> None:
        account_names = [
            requirement.account.name for requirement in accounts._get_all()
        ]
        logger.info("Trading real accounts: %s", ", ".join(account_names))

    async def _book_timeframes(
        self,
        requirements: StrategyRequirements,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timeframes: set[TimeFrame],
        notifications: Notifications,
    ) -> list[BaseExchangeAction]:
        """Book the closing timeframes once, on the newest moment the candles
        reach.

        A closing timeframe whose newest candle closed before that moment is
        left out, since the cycle that ran at its close booked it.
        """
        snapshot = next(ohlcvs._iter_timeframes(start_idx=-1))
        triggered = [
            timeframe
            for timeframe in snapshot.triggered_timeframes
            if timeframe in timeframes
        ]
        missing = timeframes.difference(triggered)
        if missing:
            logger.warning(
                "No candle closing at %s was fetched for %s, not booked",
                snapshot.timestamp,
                ", ".join(sorted(missing)),
            )
        if not triggered:
            return []
        return await self._process_timestamp(
            requirements,
            ohlcvs,
            account_snapshots,
            snapshot.timestamp,
            triggered,
            notifications,
        )

    async def _process_timestamp(
        self,
        requirements: StrategyRequirements,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        triggered_timeframes: list[TimeFrame],
        notifications: Notifications,
    ) -> list[BaseExchangeAction]:
        """Process a single timestamp across all triggered timeframes."""
        bookkeeper = BookKeeper()

        try:
            self._strategy.book_trading_actions(
                ohlcvs,
                account_snapshots,
                timestamp,
                bookkeeper,
                triggered_timeframes,
            )
            sweep_orphan_protections(
                requirements.account, account_snapshots, bookkeeper
            )
        except ExchangeRecoverableError as e:
            logger.warning(f"Exchange rejected in action generation: {e}")
        except Exception as e:
            logger.error(f"Internal error in action generation: {e}")

        post_execution_callbacks = await execute_trading_actions(
            bookkeeper.list_actions(),
            on_order_filled=[notifications.record_fill],
            on_order_placed=[notifications.record_placement],
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
            declared_waits=bookkeeper.declared_waits(),
        )

        if post_execution_callbacks:
            await _invoke_callbacks(post_execution_callbacks)
        return bookkeeper.list_actions()

    async def _notify_entry_fills(
        self,
        requirements: StrategyRequirements,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timeframes: set[TimeFrame],
        notifications: Notifications,
    ) -> None:
        """Record the entry orders the exchange fired during the closed candles.

        These orders fill between two runs, so no executed action reported
        them. Reconciling after the booking keeps the read off the path that
        places orders.
        """
        declared = [
            requirement
            for requirement in requirements.account._get_all()
            if requirement.notify_entry_fills
        ]
        if not declared:
            return

        windows = candle_windows(requirements.ohlcv, ohlcvs, timeframes)
        since = min(window.start for window in windows.values())
        for requirement in declared:
            try:
                fills = await requirement.account._executions_since(
                    since,
                    requirement.symbols,
                    max_attempts=self._max_attempts,
                    base_delay=self._base_delay,
                )
            except ExchangeRecoverableError as e:
                logger.warning("Entry fill reconciliation failed: %s", e)
                continue
            await self._record_entry_fills(
                fills,
                windows,
                requirement.describe_fills,
                account_snapshots._effects_by_execution_id(requirement.account),
                notifications,
            )

    async def _record_entry_fills(
        self,
        fills: Iterable[Execution],
        windows: CandleWindows,
        describe: FillDescriber | None,
        effects: dict[str, FillEffect],
        notifications: Notifications,
    ) -> None:
        """Record the fills of the closed candles, dropping the older ones.

        A describer that returns nothing disowns the fill, which the venue
        reports for every limit or trigger order on a declared symbol, so it
        goes unreported. A claimed fill was fired by an entry the strategy
        left resting, and its effect comes from the account's own attributed
        stream, which a read of the fills alone cannot supply.
        """
        for fill in fills:
            if fill.kind not in RESTING_ENTRY_KINDS:
                continue
            window = windows.get(fill.symbol)
            if window is None or not window.contains(fill.timestamp):
                continue
            reason = describe(fill) if describe is not None else None
            if describe is not None and reason is None:
                continue
            await _record_reconciled_fill(
                notifications,
                fill,
                kind=fill.kind,
                effect=effects.get(fill.execution_id),
                source="strategy",
                reason=reason,
            )

    async def _notify_venue_fired_fills(
        self,
        requirements: StrategyRequirements,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timeframes: set[TimeFrame],
        notifications: Notifications,
    ) -> None:
        """Record the fills the venue fired itself during the closed candles.

        A stop-loss, a take-profit and a liquidation all close a position on
        the venue's own initiative, so no booked action reported them. A
        configured fill subscriber is what asks for them, on every account
        trading a declared symbol, and a strategy declaring executions for
        its own reading asks for them too.

        The account snapshot already holds these fills from before booking.
        One firing during booking is windowed into a later cycle instead,
        since every snapshot looks back far enough to still catch it there.
        """
        declared = [
            requirement
            for requirement in requirements.account._get_all()
            if requirement.needs_executions(self._report_fills)
        ]
        if not declared:
            return

        windows = candle_windows(requirements.ohlcv, ohlcvs, timeframes)
        for requirement in declared:
            snapshot = account_snapshots.of(requirement.account)
            for symbol in requirement.symbols:
                await self._record_venue_fired_fills(
                    snapshot._venue_fired_fills(symbol),
                    windows,
                    requirement.describe_fills,
                    notifications,
                )

    async def _record_venue_fired_fills(
        self,
        fills: Iterable[Execution],
        windows: CandleWindows,
        describe: FillDescriber | None,
        notifications: Notifications,
    ) -> None:
        """Record the fills of the closed candles, dropping the older ones.

        The venue itself fired each of these, so what fired it follows from
        the kind of order it was, and the effect was attributed when the
        snapshot read the stream these fills come from.
        """
        for fill in fills:
            if fill.kind is None:
                continue
            window = windows.get(fill.symbol)
            if window is None or not window.contains(fill.timestamp):
                continue
            await _record_reconciled_fill(
                notifications,
                fill,
                kind=fill.kind,
                effect=fill.effect,
                source=VENUE_FIRED_SOURCES[fill.kind],
                reason=describe(fill) if describe is not None else None,
            )


async def _invoke_callbacks(callbacks: list[Callable[[], Awaitable[None]]]) -> None:
    results = await asyncio.gather(
        *[callback() for callback in callbacks],
        return_exceptions=True,
    )
    for i, result in enumerate(results):
        if isinstance(result, (ExchangeCriticalError, StrategyCriticalError)):
            raise result
        elif isinstance(result, ExchangeRecoverableError):
            logger.warning(f"Exchange rejected operation in callback {i}: {result}")
        elif isinstance(result, BaseException):
            logger.error(f"Internal error in callback {i}: {result}")


async def _record_reconciled_fill(
    notifications: Notifications,
    fill: Execution,
    *,
    kind: OrderType,
    effect: FillEffect | None,
    source: FillSource,
    reason: str | None,
) -> None:
    """The kind is passed apart from the fill: a fill only reaches here once its
    caller has settled that the venue named one.
    """
    order_fill = OrderFill(
        order_id=fill.order_id,
        symbol=fill.symbol,
        side=fill.side,
        kind=kind,
        quantity=fill.quantity,
        timestamp=fill.timestamp,
        realised_profit=fill.realised_profit,
        client_order_id=fill.client_order_id,
        reason=reason,
        effect=effect,
        source=source,
    )
    logger.info("Order filled | %s", order_fill)
    await notifications.record_fill(order_fill)
