import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from robottraderslab._core import (
    AccountRequirements,
    Currency,
    FillDescriber,
    OHLCVProviderProtocol,
    OHLCVRequirement,
    Symbol,
    TimeFrame,
    TimeframeSnapshot,
    assemble_all_ohlcvs_from_requirements,
    execute_trading_actions,
    fetch_account_snapshots,
    hold_margin_targets,
    log_strategy_slots,
    sweep_orphan_protections,
    to_seconds,
    widen_tracked_positions,
)
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    StrategyCriticalError,
)
from robottraderslab.strategies import (
    BookKeeper,
    OHLCVs,
    StrategyProtocol,
    StrategyRequirements,
)

from .backtest_outputs import BacktestOutputs
from .simulator import BaseSimulationEngine, CacheFillRecorder, CurrencyConverter

logger = logging.getLogger(__name__)


class Backtester:
    """Replay a strategy over every timeframe it declares, in one pass."""

    def __init__(
        self,
        strategy: StrategyProtocol,
        simulation_engine: BaseSimulationEngine,
        ohlcv_provider: OHLCVProviderProtocol,
        fill_recorder: CacheFillRecorder,
        start_date: str,
        end_date: str,
        bot_config: BotConfig | None = None,
        config_dir: Path | None = None,
        requirements_factory: Callable[[], StrategyRequirements] = StrategyRequirements,
    ):
        """
        Args:
            start_date: Backtest start date (ISO format).
            end_date: Backtest end date (ISO format).
            requirements_factory: Builds the requirements `setup` declares
                into, fresh for the run.
        """
        self._strategy = strategy
        self._simulation_engine = simulation_engine
        self._ohlcv_provider = ohlcv_provider
        self._fill_recorder = fill_recorder
        self._start_date = start_date
        self._end_date = end_date
        self._bot_config = bot_config
        self._config_dir = config_dir
        self._requirements_factory = requirements_factory
        self._start_after = datetime.fromisoformat(start_date).replace(
            tzinfo=timezone.utc
        )

    async def run(self) -> BacktestOutputs:
        """Iterates over every timestamp where at least one timeframe has
        data, so early-available symbols are not blocked by late-arriving
        ones.
        """
        self._log_data_loading_start()
        self._ohlcv_provider.set_dates(self._start_date, self._end_date)
        requirements = self._requirements_factory()
        await self._strategy.setup(requirements)
        log_strategy_slots(self._strategy)
        self._require_conversion_pairs(requirements)
        if _declares_executions(requirements.account):
            self._simulation_engine.enable_execution_recording()
        describer = _fill_describer(requirements.account)
        if describer is not None:
            self._simulation_engine.describe_fills_with(describer)
        ohlcvs = await assemble_all_ohlcvs_from_requirements(
            self._ohlcv_provider, requirements.ohlcv
        )
        ohlcvs._warn_if_data_starts_late(self._start_after)
        self._seed_last_prices(ohlcvs, requirements)
        self._simulation_engine.record_opening_equity(self._start_after)
        self._log_backtest_starting()
        self._strategy.generate_trading_signals(ohlcvs)

        executions_window = timedelta(
            seconds=_executions_window_seconds(requirements.ohlcv._get_all())
        )
        previous_timestamp: datetime | None = None
        for timeframe_snapshot in ohlcvs._iter_timeframes(after=self._start_after):
            fills_since = timeframe_snapshot.timestamp - executions_window
            await self._process_timestamp(
                requirements,
                ohlcvs,
                fills_since,
                fills_since if previous_timestamp is None else previous_timestamp,
                timeframe_snapshot,
            )
            previous_timestamp = timeframe_snapshot.timestamp

        self._simulation_engine.record_closing_equity()
        self._log_backtest_summary()
        return self._build_outputs()

    async def _process_timestamp(
        self,
        requirements: StrategyRequirements,
        ohlcvs: OHLCVs,
        fills_since: datetime,
        reported_since: datetime,
        timeframe_snapshot: TimeframeSnapshot,
    ) -> None:
        self._log_timestamp_header(timeframe_snapshot)
        self._simulation_engine.simulate_on_current_ohlcvs(
            timeframe_snapshot.timestamp, timeframe_snapshot.ohlcvs_by_symbol
        )

        bookkeeper = BookKeeper()
        account_snapshots = await fetch_account_snapshots(
            requirements.account,
            executions_since=fills_since,
        )
        await hold_margin_targets(requirements.account, account_snapshots)
        if requirements.tracker.refresh_all(
            account_snapshots, reported_since, timeframe_snapshot.timestamp
        ):
            await widen_tracked_positions(
                requirements,
                account_snapshots,
                executions_since=fills_since,
                reported_since=reported_since,
                reported_until=timeframe_snapshot.timestamp,
            )

        self._strategy.book_trading_actions(
            ohlcvs,
            account_snapshots,
            timeframe_snapshot.timestamp,
            bookkeeper,
            timeframe_snapshot.triggered_timeframes,
        )
        sweep_orphan_protections(requirements.account, account_snapshots, bookkeeper)

        post_execution_callbacks = await execute_trading_actions(
            bookkeeper.list_actions()
        )
        if post_execution_callbacks:
            await _invoke_callbacks(post_execution_callbacks)

        self._log_current_state()

    def _require_conversion_pairs(self, requirements: StrategyRequirements) -> None:
        """Register conversion rates and request the market data backing them.

        Symbols quoted in a currency other than the account currency need a
        conversion rate at every fill. Pairs margined in their base currency
        convert through their own price; anything else converts through a
        dedicated pair whose candles are added to the requirements.
        """
        converter = self._simulation_engine.converter
        added: set[tuple[Symbol, TimeFrame]] = set()
        for requirement in requirements.ohlcv._get_all():
            if requirement.symbol.quote == converter.account_currency:
                continue

            via_pair = self._conversion_pair_for(requirement, converter)
            key = (via_pair, requirement.timeframe)
            if via_pair != requirement.symbol and key not in added:
                requirements.ohlcv.add(via_pair, requirement.timeframe)
                added.add(key)

    def _conversion_pair_for(
        self, requirement: OHLCVRequirement, converter: CurrencyConverter
    ) -> Symbol:
        """Return the pair converting the requirement's quote, registering it once."""
        symbol = requirement.symbol
        registered = converter.registered_pair(symbol.quote)
        if registered is not None:
            return registered[0]

        if symbol.base == converter.account_currency:
            converter.register(symbol.quote, via=symbol, inverted=True)
            return symbol

        via_pair, inverted = self._resolve_conversion_pair(
            symbol.quote, converter, requirement.timeframe
        )
        converter.register(symbol.quote, via=via_pair, inverted=inverted)
        return via_pair

    def _seed_last_prices(
        self, ohlcvs: OHLCVs, requirements: StrategyRequirements
    ) -> None:
        """Prime the engine with the last close known on the first candle the
        backtest decides on, the first to close after its start.

        A symbol whose first candle is a gap (e.g. a forex holiday) records no
        price when that candle is simulated, and cross-currency conversion
        would have no rate to size the first orders with. When no close
        precedes the start either, the first available close is used; it
        belongs to the first candle the strategy observes anyway.
        """
        start = np.datetime64(self._start_after.replace(tzinfo=None))
        prices: dict[Symbol, float] = {}
        for requirement in requirements.ohlcv._get_all():
            symbol, timeframe = requirement.symbol, requirement.timeframe
            closes = ohlcvs.column(symbol, timeframe, "close")
            first_decided = int(
                ohlcvs.timestamps(symbol, timeframe).searchsorted(start, side="right")
            )
            preceding = closes[: first_decided + 1]
            known = preceding[~np.isnan(preceding)]
            if known.size > 0:
                prices[symbol] = float(known[-1])
                continue
            opening = closes[~np.isnan(closes)]
            if opening.size > 0:
                prices[symbol] = float(opening[0])
        self._simulation_engine.record_prices(prices)

    def _resolve_conversion_pair(
        self, currency: Currency, converter: CurrencyConverter, timeframe: TimeFrame
    ) -> tuple[Symbol, bool]:
        failures = []
        for candidate, inverted in converter.conversion_candidates(currency):
            try:
                candles = self._ohlcv_provider.fetch_ohlcv(candidate, timeframe)
            except Exception as e:
                logger.debug(f"Conversion pair {candidate} unavailable: {e}")
                failures.append(f"{candidate}: {e}")
                continue
            if not candles.empty:
                logger.info(
                    f"Converting {currency} to {converter.account_currency} "
                    f"using {candidate} candles"
                )
                return candidate, inverted
            failures.append(f"{candidate}: no data")

        failure_details = "; ".join(failures)
        raise StrategyCriticalError(
            f"No conversion data available from {currency} to "
            f"{converter.account_currency}: {failure_details}"
        )

    def _format_balances(self) -> str:
        return str(
            {
                asset: f"{balance.total:.2f}"
                for asset, balance in self._simulation_engine.get_balances().items()
            }
        )

    def _build_outputs(self) -> BacktestOutputs:
        final_balances = {
            currency: balance.total
            for currency, balance in self._simulation_engine.get_balances().items()
        }
        return BacktestOutputs(
            final_balance=final_balances,
            daily_equity_snapshots=self._simulation_engine.get_daily_equity_snapshots(),
            trade_equity_snapshots=self._simulation_engine.get_trade_equity_snapshots(),
            equity_currency=self._simulation_engine.equity_currency,
            fills=self._fill_recorder.get_fills(),
            ohlcv_provider=self._ohlcv_provider,
            bot_config=self._bot_config,
            config_dir=self._config_dir,
        )

    def _log_backtest_starting(self) -> None:  # pragma: no cover
        logger.info(f"Backtest starting | Initial Balances: {self._format_balances()}")

    def _log_data_loading_start(self) -> None:  # pragma: no cover
        logger.info("Loading data...")

    def _log_backtest_summary(self) -> None:  # pragma: no cover
        logger.info(f"Backtest ended | Final Balances: {self._format_balances()}")
        self._log_open_positions()

    def _log_timestamp_header(
        self, timeframe_snapshot: TimeframeSnapshot
    ) -> None:  # pragma: no cover
        logger.debug("=" * 50)
        logger.debug(
            f"Timestamp: {timeframe_snapshot.timestamp} | triggered timeframes: {', '.join(timeframe_snapshot.triggered_timeframes)}"
        )

    def _log_current_state(self) -> None:  # pragma: no cover
        logger.debug("-" * 50)
        logger.debug(f"Current Balances: {self._format_balances()}")
        self._log_open_positions(logging.DEBUG)

    def _log_open_positions(
        self, level: int = logging.INFO
    ) -> None:  # pragma: no cover
        positions = self._simulation_engine.open_positions
        if positions:
            logger.log(level, "Open Positions:")
            for position in positions.values():
                logger.log(
                    level,
                    f"  {position.symbol}: qty={position.quantity:.8f}, avg_entry=${position.average_entry_price:,.2f}",
                )
        else:
            logger.log(level, "  - no open positions")


def _declares_executions(requirements: AccountRequirements) -> bool:
    """A backtest has no fill subscriber, so the need for executions can
    only come from what the account itself declared.
    """
    return any(
        requirement.needs_executions(report_fills=False)
        for requirement in requirements._get_all()
    )


def _fill_describer(requirements: AccountRequirements) -> FillDescriber | None:
    """Every account in a backtest trades the one simulated engine, which can
    only ask one describer.

    Raises:
        StrategyCriticalError: If more than one account declares a describer.
    """
    describers = [
        requirement.describe_fills
        for requirement in requirements._get_all()
        if requirement.describe_fills is not None
    ]
    if len(describers) > 1:
        raise StrategyCriticalError(
            f"{len(describers)} accounts declare describe_fills; a backtest "
            "simulates one account, so declare it on one account only."
        )
    return describers[0] if describers else None


def _executions_window_seconds(requirements: Iterable[OHLCVRequirement]) -> int:
    """Return, in seconds, the largest lookback any requirement declares.

    A live cycle loads only its declared lookback, so its oldest candle is
    always that close to the current one; replaying the same span in a
    backtest keeps the executions a strategy sees bounded by what it would
    have had live, whatever the backtest's own length.
    """
    return max(
        (
            (requirement.lookback - 1) * to_seconds(requirement.timeframe)
            for requirement in requirements
            if requirement.lookback > 0
        ),
        default=0,
    )


async def _invoke_callbacks(callbacks: list[Callable[[], Awaitable[None]]]) -> None:
    results = await asyncio.gather(
        *[callback() for callback in callbacks],
        return_exceptions=True,
    )
    for i, result in enumerate(results):
        if isinstance(result, (ExchangeCriticalError, StrategyCriticalError)):
            raise result
        elif isinstance(result, BaseException):
            logger.error(f"Failed to invoke callback {i}: {result}")
