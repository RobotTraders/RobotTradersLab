import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial

from robottraderslab._internal import (
    DryRunFailure,
    assemble_available_ohlcvs_from_requirements,
    assume_margin_targets,
    clean_removed_profiles,
    collect_profile_failures,
    confirm_booked_protections,
    dropped_pair_report,
    failure_of,
    hold_margin_targets,
    log_dropped_pair,
    log_strategy_slots,
    sitting_out_symbols,
    sweep_orphan_protections,
    widen_tracked_positions,
)
from robottraderslab.exceptions import (
    BotCriticalError,
    ExchangeCriticalError,
    ExchangeRecoverableError,
)
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    OHLCVs,
    StrategyProtocol,
    StrategyRequirements,
)
from robottraderslab_core import Symbol, TimeFrame
from robottraderslab_core.engine import (
    BASE_DELAY_SECONDS,
    MAX_ATTEMPTS,
    RESTING_ENTRY_KINDS,
    VENUE_FIRED_SOURCES,
    AccountProtocol,
    Notifications,
    OnOrderFilled,
    OnOrderPlaced,
    execute_trading_actions,
    fetch_account_snapshots,
    read_instruments,
)
from robottraderslab_core.exchanges import (
    BaseExchangeAction,
    Execution,
    FillEffect,
    FillSource,
    OHLCVProviderProtocol,
    OrderFill,
    OrderType,
    TradingStatus,
    retry_on_transient,
)
from robottraderslab_core.ohlcv import (
    CandleWindows,
    TimeframeSnapshot,
    candle_closes_at,
    candle_windows,
    to_seconds,
)
from robottraderslab_core.strategies import FillDescriber

from .booked_fills import BookedFills
from .exit_only import restrict_to_exit_only
from .halts import held_back, log_halts, restrict_halted, without_actions_on

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """The moment a cycle runs at, by the wall clock."""
    return datetime.now(UTC)


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
        clean_removed_profiles: bool = True,
        exit_only_symbols: frozenset[Symbol] = frozenset(),
        dry_run: bool = False,
        clock: Callable[[], datetime] = utc_now,
        requirements_factory: Callable[[], StrategyRequirements] = StrategyRequirements,
    ) -> None:
        """Initialise a multi-timeframe live trading bot.

        Args:
            on_fill: Notified once for each fill of a cycle, whatever told
                the engine about it. Subscribing one is also what puts a
                stop-loss, take-profit or liquidation the venue fired within
                reach, on every account trading a declared symbol.
            on_placement: Notified once for each order the venue accepted.
            max_attempts: Total retry attempts for transient errors.
            base_delay: Base delay in seconds (doubled on each retry).
            clean_removed_profiles: Off, each cycle names what it would have
                cleaned and leaves it.
            exit_only_symbols: The symbols run in exit-only mode: the cycle
                reads their position and open orders, sends only what brings
                the position down of what the strategy books on them, and
                cancels the resting orders that could open, increase or flip
                it. Every other symbol trades as usual, except a symbol its
                venue reports closed to new positions, which the cycle runs
                in exit-only mode as well, and one closed to every order,
                which it sends nothing on.
            dry_run: Runs the cycle to see whether it runs through and sends
                nothing: no order, cancel, leverage or margin-mode setting and
                no notification. It runs `setup`, reads the candles and the
                account state, and calls the strategy's signal slots and
                booking slots with every declared timeframe as triggered,
                whether its candle closes at the cycle's moment or not, on a
                snapshot holding the declared margin targets and a book it
                discards. Each failure it meets is reported by `failures` in
                place of being logged.
            clock: The cycle's moment, read once as the cycle starts. A caller
                running a dry run and the cycle that follows it hands both the
                same moment, so that both book the candles that closed when
                the first began.
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
        self._clean_removed_profiles = clean_removed_profiles
        self._exit_only_symbols = exit_only_symbols
        self._dry_run = dry_run
        self._clock = clock
        self._log = _CycleLog(dry_run)
        self._exit_only_halts: dict[AccountProtocol, frozenset[Symbol]] = {}
        self._closed_halts: dict[AccountProtocol, frozenset[Symbol]] = {}

    @property
    def failures(self) -> list[DryRunFailure]:
        """What a dry run met, in the order it met it.

        The failures of each section the build refused, of each profile slot
        that did not run through and of each step of the cycle that failed.
        Empty unless the bot was told to dry run.
        """
        return self._log.failures

    async def run(self) -> None:
        """Run one complete trading cycle.

        A dry run sends nothing and records each failure it meets in
        `failures`, a `BotCriticalError` included, since no bot is left to
        stop.

        Raises:
            BotCriticalError: If the bot cannot run, outside a dry run.
            ExchangeCriticalError: If the venue cannot be traded on.
        """
        if self._dry_run:
            await self._run_dry()
            return
        self._log_cycle_header()
        await self._run_cycle()

    async def _book_removed_profile_cleanup(
        self,
        requirements: StrategyRequirements,
        account_snapshots: AccountSnapshots,
        bookkeeper: BookKeeper,
    ) -> None:
        """Clean up after the removed profiles, skipped when its reads fail.

        The cleanup is no part of the strategy's own booking, so a failure of
        its reads costs the cleanup of this cycle and nothing else.
        """
        try:
            await clean_removed_profiles(
                requirements.account,
                account_snapshots,
                bookkeeper,
                sitting_out=sitting_out_symbols(self._strategy),
                clean=self._clean_removed_profiles,
                max_attempts=self._max_attempts,
                base_delay=self._base_delay,
            )
        except ExchangeRecoverableError as e:
            logger.warning(f"Cleanup of removed profiles skipped: {e}")
        except Exception as e:
            logger.error(f"Internal error cleaning removed profiles, skipped: {e}")

    async def _book_timeframes(
        self,
        requirements: StrategyRequirements,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timeframes: set[TimeFrame],
        notifications: Notifications,
        booked_fills: BookedFills,
    ) -> list[BaseExchangeAction]:
        """Book the closing timeframes once, on the newest moment the candles reach.

        A closing timeframe whose newest candle closed before that moment is
        left out, since the cycle that ran at its close booked it.
        """
        snapshot = _newest_snapshot(ohlcvs)
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
            booked_fills,
        )

    def _booked_timeframes(self, declared: set[TimeFrame]) -> set[TimeFrame]:
        """The declared timeframes this cycle books, every one in a dry run.

        Otherwise those whose candle closes at the clock's moment.
        """
        if self._dry_run:
            return declared
        now = self._clock()
        return {timeframe for timeframe in declared if candle_closes_at(timeframe, now)}

    def _drop_actions_on_closed_halts(self, bookkeeper: BookKeeper) -> None:
        """Leave out every action on a symbol closed to every order.

        What the engine books itself goes too: the cleanup of a removed
        profile, the sweep of an orphan protection and the cancels of exit-only
        mode.
        """
        if not self._closed_halts:
            return
        bookkeeper._replace_actions(
            without_actions_on(self._closed_halts, bookkeeper.list_actions())
        )

    def _dry_run_booking(
        self,
        requirements: StrategyRequirements,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timeframes: Iterable[TimeFrame],
    ) -> None:
        """Call the strategy's booking once with every timeframe triggered.

        The book is one nothing reads. A profile's booking runs only on the
        cycles its timeframe closes on, and a dry run must see all of it run.
        """
        self._generate_actions(
            requirements,
            ohlcvs,
            account_snapshots,
            _newest_snapshot(ohlcvs).timestamp,
            sorted(timeframes, key=to_seconds),
            BookKeeper(),
        )

    async def _execute_booking(
        self,
        requirements: StrategyRequirements,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        closing_timeframes: set[TimeFrame],
        notifications: Notifications,
        reported_since: datetime,
    ) -> list[OrderFill]:
        """Book the closing timeframes, send what is booked, confirm its protections.

        Returns:
            The fills the venue settled.
        """
        booked_fills = BookedFills(
            requirements.account,
            reported_since,
            settling=self._report_fills,
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
        )
        try:
            executed = await self._book_timeframes(
                requirements,
                ohlcvs,
                account_snapshots,
                closing_timeframes,
                notifications,
                booked_fills,
            )
            await confirm_booked_protections(
                executed, max_attempts=self._max_attempts, base_delay=self._base_delay
            )
            return await booked_fills.settled()
        finally:
            booked_fills.cancel()

    async def _fetch_ohlcvs(self, requirements: StrategyRequirements) -> OHLCVs | None:
        return await assemble_available_ohlcvs_from_requirements(
            self._ohlcv_provider,
            requirements.ohlcv,
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
            on_dropped=self._log.dropped_pair,
        )

    def _generate_actions(
        self,
        requirements: StrategyRequirements,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        triggered_timeframes: list[TimeFrame],
        bookkeeper: BookKeeper,
    ) -> None:
        """Book the strategy's actions, then sweep the orphan protections.

        A dry run books and leaves the sweep, which would cancel orders.
        """
        try:
            self._strategy.book_trading_actions(
                ohlcvs,
                account_snapshots,
                timestamp,
                bookkeeper,
                triggered_timeframes,
            )
            if not self._dry_run:
                sweep_orphan_protections(
                    requirements.account, account_snapshots, bookkeeper
                )
        except ExchangeRecoverableError as e:
            self._log.warning(f"Exchange rejected in action generation: {e}")
        except Exception as e:
            self._log.error(f"Internal error in action generation: {e}")

    def _generate_signals(self, ohlcvs: OHLCVs) -> bool:
        """Compute the strategy's signals, False when they failed."""
        try:
            self._strategy.generate_trading_signals(ohlcvs)
        except Exception as e:
            self._log.error(
                f"Signal generation failed, skipping cycle: {e}",
                recoverable=isinstance(e, ExchangeRecoverableError),
            )
            return False
        return True

    def _hold_halts(self, requirements: StrategyRequirements) -> None:
        """Hold back each symbol an account's venue halted, on that account.

        A symbol closed to new positions runs in exit-only mode this cycle, and
        one closed to every order is sent nothing. The account whose listing
        reports the symbol open trades it as usual.
        """
        halts = requirements.account._halts()
        log_halts(halts, logging.DEBUG if self._dry_run else logging.WARNING)
        self._exit_only_halts = held_back(
            requirements.account,
            halts,
            TradingStatus.CLOSED_TO_NEW_POSITIONS,
        )
        self._closed_halts = held_back(
            requirements.account, halts, TradingStatus.CLOSED_TO_EVERY_ORDER
        )

    async def _hold_margin_targets(
        self, requirements: StrategyRequirements, account_snapshots: AccountSnapshots
    ) -> None:
        """Set the declared margin targets on the venue.

        A dry run writes them on the snapshot only.
        """
        if self._dry_run:
            assume_margin_targets(
                requirements.account, account_snapshots, closed=self._closed_halts
            )
            return
        await hold_margin_targets(
            requirements.account,
            account_snapshots,
            closed=self._closed_halts,
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
        )

    async def _load_strategy(self) -> StrategyRequirements:
        """Load strategy with fresh requirements to avoid duplicates on retry."""
        requirements = self._requirements_factory()
        await self._strategy.setup(requirements)
        await read_instruments(requirements.account)
        return requirements

    def _log_cycle_header(self) -> None:  # pragma: no cover
        logger.info("=" * 80)
        logger.info("LiveBot Trading Cycle Started")

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

    async def _process_timestamp(
        self,
        requirements: StrategyRequirements,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        triggered_timeframes: list[TimeFrame],
        notifications: Notifications,
        booked_fills: BookedFills,
    ) -> list[BaseExchangeAction]:
        """Process a single timestamp across all triggered timeframes."""
        bookkeeper = BookKeeper()

        await self._book_removed_profile_cleanup(
            requirements, account_snapshots, bookkeeper
        )
        engine_actions = (
            bookkeeper.list_actions()
            if self._exit_only_symbols or self._exit_only_halts
            else ()
        )
        self._generate_actions(
            requirements,
            ohlcvs,
            account_snapshots,
            timestamp,
            triggered_timeframes,
            bookkeeper,
        )
        self._restrict_to_exit_only(
            requirements, account_snapshots, bookkeeper, engine_actions
        )
        self._drop_actions_on_closed_halts(bookkeeper)
        await self._send_actions(bookkeeper, notifications, booked_fills)
        return bookkeeper.list_actions()

    async def _read_account_state(
        self, requirements: StrategyRequirements, ohlcvs: OHLCVs
    ) -> AccountSnapshots | None:
        """Read the accounts' state.

        Returns:
            The snapshots, None when the read failed and the cycle is skipped.
        """
        self._hold_halts(requirements)
        try:
            account_snapshots = await fetch_account_snapshots(
                requirements.account._widened_for_exit_only(
                    self._exit_only_symbols, self._exit_only_halts
                ),
                executions_since=ohlcvs._earliest_candle_open(),
                report_fills=self._report_fills,
                max_attempts=self._max_attempts,
                base_delay=self._base_delay,
            )
        except ExchangeRecoverableError as e:
            self._log.warning(f"Account state fetch failed, skipping cycle: {e}")
            return None
        return account_snapshots

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

    def _restrict_to_exit_only(
        self,
        requirements: StrategyRequirements,
        account_snapshots: AccountSnapshots,
        bookkeeper: BookKeeper,
        engine_actions: Sequence[BaseExchangeAction],
    ) -> None:
        """The actions of a booking that raised are restricted as well."""
        if not self._exit_only_symbols and not self._exit_only_halts:
            return
        booked = bookkeeper.list_actions()[len(engine_actions) :]
        allowed = restrict_to_exit_only(
            requirements.account, account_snapshots, self._exit_only_symbols, booked
        )
        allowed = restrict_halted(
            requirements.account, account_snapshots, self._exit_only_halts, allowed
        )
        bookkeeper._replace_actions([*engine_actions, *allowed])

    async def _run_cycle(self) -> None:
        requirements = await self._set_up_strategy()
        declared_timeframes = {r.timeframe for r in requirements.ohlcv._get_all()}
        closing_timeframes = self._booked_timeframes(declared_timeframes)
        if declared_timeframes and not closing_timeframes:
            logger.info("No declared candle closes this minute, nothing to do")
            return

        ohlcvs = await self._fetch_ohlcvs(requirements)
        if ohlcvs is None:
            self._log.error(
                "No declared market data could be fetched, skipping cycle",
                recoverable=True,
            )
            return

        closing_timeframes = _timeframes_held(ohlcvs, closing_timeframes)
        if not closing_timeframes or not self._generate_signals(ohlcvs):
            return
        account_snapshots = await self._read_account_state(requirements, ohlcvs)
        if account_snapshots is None:
            return
        await self._hold_margin_targets(requirements, account_snapshots)
        await self._settle_and_book(
            requirements, ohlcvs, account_snapshots, closing_timeframes
        )

    async def _run_dry(self) -> None:
        try:
            with collect_profile_failures(self._strategy) as failures:
                self._log.failures = failures
                await self._run_cycle()
        except ExchangeCriticalError:
            raise
        except (Exception, BotCriticalError) as error:
            logger.debug("A dry run met %r", error, exc_info=True)
            self._log.failures.append(failure_of(error, None))

    async def _send_actions(
        self,
        bookkeeper: BookKeeper,
        notifications: Notifications,
        booked_fills: BookedFills,
    ) -> None:
        """Send the booked actions, then run what must follow their execution."""
        post_execution_callbacks = await execute_trading_actions(
            bookkeeper.list_actions(),
            on_order_filled=[booked_fills.record_fill],
            on_order_placed=[
                notifications.record_placement,
                booked_fills.record_placement,
            ],
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
            declared_waits=bookkeeper.declared_waits(),
        )
        if post_execution_callbacks:
            await _invoke_callbacks(post_execution_callbacks)

    async def _send_booking(
        self,
        requirements: StrategyRequirements,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        closing_timeframes: set[TimeFrame],
        reported_since: datetime,
    ) -> None:
        """Book the closing timeframes, send the booking and notify its fills."""
        notifications = Notifications(self._on_fill, self._on_placement)
        settled_fills = await self._execute_booking(
            requirements,
            ohlcvs,
            account_snapshots,
            closing_timeframes,
            notifications,
            reported_since,
        )
        for order_fill in settled_fills:
            await notifications.record_fill(order_fill)
        await self._notify_entry_fills(
            requirements, ohlcvs, account_snapshots, closing_timeframes, notifications
        )
        await self._notify_venue_fired_fills(
            requirements, ohlcvs, account_snapshots, closing_timeframes, notifications
        )
        await notifications.flush()

    async def _set_up_strategy(self) -> StrategyRequirements:
        """Run `setup`, retrying a transient error, and log the slots it fills."""
        requirements = await retry_on_transient(
            self._load_strategy,
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
        )
        log_strategy_slots(self._strategy)
        return requirements

    async def _settle_and_book(
        self,
        requirements: StrategyRequirements,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        closing_timeframes: set[TimeFrame],
    ) -> None:
        """Settle the tracked positions, then book and send the closing timeframes.

        A dry run books them and sends nothing.
        """
        closed = candle_windows(requirements.ohlcv, ohlcvs, closing_timeframes)
        reported_since = min(window.start for window in closed.values())
        reported_until = max(window.end for window in closed.values())
        if not await self._settle_tracker(
            requirements, ohlcvs, account_snapshots, reported_since, reported_until
        ):
            return
        if self._dry_run:
            self._dry_run_booking(
                requirements, ohlcvs, account_snapshots, closing_timeframes
            )
            return
        await self._send_booking(
            requirements, ohlcvs, account_snapshots, closing_timeframes, reported_since
        )

    async def _settle_tracker(
        self,
        requirements: StrategyRequirements,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        reported_since: datetime,
        reported_until: datetime,
    ) -> bool:
        """Refresh the tracked positions, False when it failed."""
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
            self._log.warning(
                f"Tracked position settlement failed, skipping cycle: {e}"
            )
            return False
        except Exception as e:
            self._log.error(f"Internal error refreshing tracked positions: {e}")
            return False
        return True


def _newest_snapshot(ohlcvs: OHLCVs) -> TimeframeSnapshot:
    return next(ohlcvs._iter_timeframes(start_idx=-1))


async def _record_reconciled_fill(
    notifications: Notifications,
    fill: Execution,
    *,
    kind: OrderType,
    effect: FillEffect | None,
    source: FillSource,
    reason: str | None,
) -> None:
    """A fill only reaches here once its caller has settled that the venue named
    its kind. Every connector trades linear contracts only. A linear
    contract's price is quoted per unit of the base currency, so price times
    quantity is the fill's value in the quote currency.
    """
    order_fill = OrderFill(
        order_id=fill.order_id,
        symbol=fill.symbol,
        side=fill.side,
        kind=kind,
        quantity=fill.quantity,
        timestamp=fill.timestamp,
        filled_value=fill.price * fill.quantity,
        realised_profit=fill.realised_profit,
        client_order_id=fill.client_order_id,
        reason=reason,
        effect=effect,
        source=source,
    )
    logger.info("Order filled | %s", order_fill)
    await notifications.record_fill(order_fill)


def _timeframes_held(ohlcvs: OHLCVs, timeframes: Iterable[TimeFrame]) -> set[TimeFrame]:
    return {timeframe for timeframe in timeframes if ohlcvs._has_timeframe(timeframe)}


async def _invoke_callbacks(callbacks: Iterable[Callable[[], Awaitable[None]]]) -> None:
    results = await asyncio.gather(
        *[callback() for callback in callbacks],
        return_exceptions=True,
    )
    for i, result in enumerate(results):
        if isinstance(result, (ExchangeCriticalError, BotCriticalError)):
            raise result
        elif isinstance(result, ExchangeRecoverableError):
            logger.warning(f"Exchange rejected operation in callback {i}: {result}")
        elif isinstance(result, BaseException):
            logger.error(f"Internal error in callback {i}: {result}")


@dataclass
class _CycleLog:
    """What the cycle reports, logged as it happens or held for a dry run.

    A dry run reports each failure to its caller, which words them once the
    run ends, so no line reaches the log or a notifier twice.
    """

    dry_run: bool
    failures: list[DryRunFailure] = field(default_factory=list)

    def dropped_pair(
        self, symbol: Symbol, timeframe: TimeFrame, outcome: BaseException
    ) -> None:
        """Report a pair the fetch dropped, a recoverable failure of a dry run.

        The profiles of its symbol are left unverified on it.
        """
        _, sentence = dropped_pair_report(symbol, timeframe, outcome)
        self._report(
            symbol,
            sentence,
            True,
            partial(log_dropped_pair, symbol, timeframe, outcome),
        )

    def error(self, message: str, *, recoverable: bool = False) -> None:
        """Report a failure of the cycle itself, recoverable when the caller says so."""
        self._report(None, message, recoverable, partial(logger.error, message))

    def warning(self, message: str) -> None:
        """Report a recoverable failure of the cycle, which the venue caused."""
        self._report(None, message, True, partial(logger.warning, message))

    def _report(
        self,
        symbol: Symbol | None,
        message: str,
        recoverable: bool,
        log: Callable[[], None],
    ) -> None:
        if not self.dry_run:
            log()
            return
        logger.debug(message)
        self.failures.append(DryRunFailure(symbol, message, recoverable))
