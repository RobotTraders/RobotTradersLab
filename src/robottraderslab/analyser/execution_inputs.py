import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime

from robottraderslab._core import (
    Balance,
    Currency,
    Execution,
    OHLCVProviderProtocol,
    PositionSnapshot,
    Symbol,
)
from robottraderslab.bootstrap import ReportConfig
from robottraderslab.exceptions import StrategyCriticalError

from .analysis_inputs import AnalysisInputs
from .execution_equity import build_equity_curve, recover_opening_balance
from .execution_positions import build_open_positions
from .execution_trades import PairedTrades, build_trades
from .reference_price_handler import resolve_reference

logger = logging.getLogger(__name__)


def build_execution_inputs(
    *,
    executions: Sequence[Execution],
    positions: Mapping[Symbol, PositionSnapshot],
    balances: Mapping[Currency, Balance],
    symbols: Iterable[Symbol],
    since: datetime,
    until: datetime,
    ohlcv_provider: OHLCVProviderProtocol,
    report_config: ReportConfig,
) -> AnalysisInputs:
    """
    Args:
        executions: Every execution to read, reaching back far enough to
            establish what was already open at `since` where the venue
            allows it, and covering the window itself.
        positions: The venue's open positions.
        balances: What the venue states the account holds now. Every share
            of a balance goes unstated where the settlement currency has no
            entry.
        symbols: The symbols the report covers. They must settle in one
            currency, since combining their executions into one curve
            otherwise mixes amounts stated in different currencies.
        since: Start of the reported window.
        until: End of the reported window.
        ohlcv_provider: Candle source for the reference price.

    Raises:
        StrategyCriticalError: If the symbols settle in more than one currency.
    """
    settlement = _settlement_currency(symbols)
    paired = build_trades(executions, since=since, until=until)
    _warn_about_partial_pairing(paired)

    window_executions = _executions_since(executions, since)
    opening_balance = _opening_balance(balances, settlement, window_executions)
    equity_curve = build_equity_curve(
        window_executions,
        since=since,
        until=until,
        opening_balance=opening_balance,
    )
    reference_price, reference_symbol_str = resolve_reference(
        report_config.reference_symbol,
        report_config.reference_timeframe,
        equity_curve.index,
        ohlcv_provider,
    )
    return AnalysisInputs(
        trades=paired.trades,
        open_positions=build_open_positions(positions),
        equity_curve=equity_curve,
        initial_balance=opening_balance,
        reference_price=reference_price,
        reference_symbol_str=reference_symbol_str,
    )


def _settlement_currency(symbols: Iterable[Symbol]) -> Currency:
    settlements = {symbol.settlement for symbol in symbols}
    if len(settlements) != 1:
        raise StrategyCriticalError(
            f"The report covers symbols settling in several currencies "
            f"{sorted(settlements)}; it measures one account in one currency."
        )
    return settlements.pop()


def _warn_about_partial_pairing(paired: PairedTrades) -> None:
    if paired.unpairable:
        logger.warning(
            "%d execution(s) closed a position whose entry the venue no "
            "longer serves; the window's trades are partial.",
            paired.unpairable,
        )
    if paired.unpriced:
        logger.warning(
            "%d execution(s) closed a position the venue states no profit "
            "for; the window's trades are partial.",
            paired.unpriced,
        )


def _executions_since(
    executions: Sequence[Execution], since: datetime
) -> list[Execution]:
    return [execution for execution in executions if execution.timestamp >= since]


def _opening_balance(
    balances: Mapping[Currency, Balance],
    settlement: Currency,
    window_executions: Sequence[Execution],
) -> float | None:
    held = balances.get(settlement)
    opening_balance = (
        recover_opening_balance(held.total, window_executions)
        if held is not None
        else None
    )
    if opening_balance is None:
        logger.warning(
            "No %s balance to measure the window against; the report states "
            "its currency figures and leaves every share of a balance out.",
            settlement,
        )
    return opening_balance
