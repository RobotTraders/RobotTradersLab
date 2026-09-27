from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Literal

import pandas as pd

from robottraderslab._core import (
    ReasonTable,
    ReportGroup,
    ReportTable,
    ReportTableRow,
    SymbolAnalysis,
    format_percent,
)

from .performance_report import _ALL_COLUMN, build_trades_table
from .trade_filter import TradeFilter
from .trade_metrics import (
    TradeMetricsResult,
    calculate_enhanced_reason_metrics,
    compute_trade_metrics,
)

_EQUITY_SHARES_TITLE = "As a share of the initial equity"

type _Share = tuple[str, Callable[[TradeMetricsResult], float]]

_EQUITY_SHARES: tuple[_Share, ...] = (
    ("Average PnL", lambda metrics: metrics.avg_trade_pnl),
    ("Average winning trade", lambda metrics: metrics.avg_winning_trade_pnl),
    ("Average losing trade", lambda metrics: metrics.avg_losing_trade_pnl),
)


def build_filtered_table(
    metrics: TradeMetricsResult, initial_equity: float
) -> ReportTable:
    return _table_with_equity_shares(
        [(_ALL_COLUMN, metrics)], _filter_title(metrics.filter_applied), initial_equity
    )


def build_long_short_table(trades: pd.DataFrame, initial_equity: float) -> ReportTable:
    """
    Raises:
        ValueError: If the run took no long trade, or no short trade.
    """
    return _table_with_equity_shares(
        [
            ("Long", _side_metrics(trades, "long")),
            ("Short", _side_metrics(trades, "short")),
        ],
        "LONG vs SHORT TRADE ANALYSIS",
        initial_equity,
    )


def build_reason_tables(trades: pd.DataFrame) -> list[ReasonTable]:
    if trades.empty:
        return []

    reasons = calculate_enhanced_reason_metrics(trades)
    return [
        _reason_table("Entry Reasons", reasons.entry_reasons_table.reset_index()),
        _reason_table("Exit Reasons", reasons.exit_reasons_table.reset_index()),
        _reason_table("Entry to Exit Reason Pairs", reasons.entry_exit_pairs_table),
    ]


def build_symbol_analysis(
    trades: pd.DataFrame, symbol: str, initial_equity: float
) -> SymbolAnalysis:
    """
    Raises:
        ValueError: If the run took no trade on the symbol.
    """
    symbol_filter = TradeFilter(symbol=symbol)
    metrics = compute_trade_metrics(trades, symbol_filter)
    return SymbolAnalysis(
        symbol=symbol,
        trades=_table_with_equity_shares(
            [(_ALL_COLUMN, metrics)], f"SYMBOL ANALYSIS: {symbol}", initial_equity
        ),
        reasons=build_reason_tables(symbol_filter.apply_filter(trades)),
    )


def _table_with_equity_shares(
    measured: Sequence[tuple[str, TradeMetricsResult]],
    title: str,
    initial_equity: float,
) -> ReportTable:
    table = build_trades_table(measured, title=title)
    figures = [metrics for _, metrics in measured]
    shares = ReportGroup(
        title=_EQUITY_SHARES_TITLE,
        rows=_equity_share_rows(figures, initial_equity),
    )
    return replace(table, groups=[*table.groups, shares])


def _equity_share_rows(
    figures: Sequence[TradeMetricsResult], initial_equity: float
) -> list[ReportTableRow]:
    return [
        ReportTableRow(
            label=label,
            values=[
                format_percent(share(metrics) / initial_equity) for metrics in figures
            ],
        )
        for label, share in _EQUITY_SHARES
    ]


def _filter_title(trade_filter: TradeFilter) -> str:
    parts: list[str] = []
    if trade_filter.side is not None:
        parts.append(trade_filter.side)
    if trade_filter.symbol is not None:
        parts.append(trade_filter.symbol)
    parts.extend(trade_filter.profile_name_patterns)
    return f"FILTERED ANALYSIS: {', '.join(parts)}"


def _side_metrics(
    trades: pd.DataFrame, side: Literal["long", "short"]
) -> TradeMetricsResult:
    try:
        return compute_trade_metrics(trades, TradeFilter(side=side))
    except ValueError as e:
        raise ValueError(f"No {side} trades found") from e


def _reason_table(title: str, figures: pd.DataFrame) -> ReasonTable:
    key, *columns = (str(column) for column in figures.columns)
    return ReasonTable(
        title=title,
        key=key,
        columns=columns,
        rows=[_reason_row(*row) for row in figures.itertuples(index=False)],
    )


def _reason_row(
    key: object, count: int, share: float, win_rate: float, average: float
) -> ReportTableRow:
    return ReportTableRow(
        label=str(key),
        values=[str(count), f"{share:.1f}%", f"{win_rate:.1f}%", f"{average:+.4f}"],
    )
