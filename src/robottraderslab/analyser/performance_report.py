from collections.abc import Callable, Sequence
from datetime import datetime

from robottraderslab._core import (
    PerformanceReport,
    ReportGroup,
    ReportRow,
    ReportSection,
    ReportTable,
    ReportTableRow,
    format_money,
    format_percent,
    format_ratio,
)

from .equity_metrics import EquityMetricsResult
from .trade_metrics import TradeDrawdownMetrics, TradeMetricsResult

_ALL_COLUMN = "All"
_SIDELESS_TITLE = "Filtered"
_TRADES_TITLE = "Trades"

type _Figure = tuple[str, Callable[[TradeMetricsResult], str]]


def build_performance_report(
    trade_metrics: TradeMetricsResult,
    equity_metrics: EquityMetricsResult,
    trade_drawdown: TradeDrawdownMetrics,
    sides: Sequence[TradeMetricsResult] = (),
) -> PerformanceReport:
    """Describe how a run performed, once, for every medium that shows it.

    Args:
        trade_drawdown: The deepest fall between two trade completions.
        sides: The same run measured over one side's trades at a time, so long
            and short can be read against each other. A side the strategy never
            took is absent.
    """
    return PerformanceReport(
        headline=_headline_figures(trade_metrics, equity_metrics),
        sections=[
            _overview_section(trade_metrics, equity_metrics),
            _ratios_section(trade_drawdown, equity_metrics),
        ],
        trades=_trades_table(trade_metrics, sides),
    )


def _headline_figures(
    trade_metrics: TradeMetricsResult, equity_metrics: EquityMetricsResult
) -> list[ReportRow]:
    """Pick the few figures that answer "was this any good" on their own."""
    return _rows(
        ("PnL", _pnl(equity_metrics)),
        ("Max drawdown", _drawdown(equity_metrics)),
        ("Win rate", _win_rate(trade_metrics)),
        ("Profit factor", format_ratio(trade_metrics.profit_factor)),
        *_sharpe_headline(equity_metrics),
    )


def _sharpe_headline(metrics: EquityMetricsResult) -> tuple[tuple[str, str], ...]:
    if metrics.sharpe_ratio is None:
        return ()
    return (("Sharpe ratio", format_ratio(metrics.sharpe_ratio)),)


def _pnl(metrics: EquityMetricsResult) -> str:
    return _amount_and_share(metrics.net_profit, metrics.initial_equity)


def _drawdown(metrics: EquityMetricsResult) -> str:
    if metrics.max_drawdown is None:
        return format_money(metrics.max_drawdown_amount)
    return f"{format_money(metrics.max_drawdown_amount)} ({format_percent(metrics.max_drawdown)})"


def _amount_and_share(amount: float, balance: float | None) -> str:
    """The amount reads off the curve's own two ends and stands whatever the
    curve is measured from.

    The share needs that measure.
    """
    if balance is None or balance == 0.0:
        return format_money(amount)
    return f"{format_money(amount)} ({format_percent(amount / balance)})"


def _win_rate(metrics: TradeMetricsResult) -> str:
    won = f"{metrics.winning_trades}/{metrics.total_trades}"
    return f"{format_percent(metrics.win_rate)} ({won})"


def _overview_section(
    trade_metrics: TradeMetricsResult, equity_metrics: EquityMetricsResult
) -> ReportSection:
    return _section(
        "Overview",
        ("Period start", _moment(equity_metrics.period_start)),
        ("Period end", _moment(equity_metrics.period_end)),
        *_balance_figures(equity_metrics),
        ("PnL", _pnl(equity_metrics)),
        *_hodl_figures(equity_metrics),
        ("Time in position", format_percent(trade_metrics.time_in_position_ratio)),
    )


def _balance_figures(metrics: EquityMetricsResult) -> tuple[tuple[str, str], ...]:
    """Both come from the same source as every ratio that divides by them, so
    they are shown together and dropped together.
    """
    if metrics.initial_equity is None or metrics.final_equity is None:
        return ()
    return (
        ("Initial balance", format_money(metrics.initial_equity)),
        ("Final equity", format_money(metrics.final_equity)),
    )


def _hodl_figures(metrics: EquityMetricsResult) -> tuple[tuple[str, str], ...]:
    return tuple(
        figure
        for hodl in metrics.hodls
        for figure in (
            (f"Hodl performance ({hodl.held})", format_percent(hodl.hodl_return)),
            (
                f"Performance vs hodl ({hodl.held})",
                format_percent(hodl.performance_vs_hodl),
            ),
        )
    )


def _ratios_section(
    trade_drawdown: TradeDrawdownMetrics, equity_metrics: EquityMetricsResult
) -> ReportSection:
    return _section(
        "Risk & ratios",
        ("Max drawdown", _drawdown(equity_metrics)),
        *_trade_drawdown_figures(trade_drawdown, equity_metrics),
        *_curve_ratio_figures(equity_metrics),
    )


def _trade_drawdown_figures(
    trade_drawdown: TradeDrawdownMetrics, equity_metrics: EquityMetricsResult
) -> tuple[tuple[str, str], ...]:
    """The share divides by the peak the fall came off, so a curve measured from
    no balance of its own takes a share of an amount that is not one, and the
    fall is left to the row measuring the curve as a whole.
    """
    amount = trade_drawdown.amount
    if amount is None or equity_metrics.initial_equity is None:
        return ()
    share = format_percent(trade_drawdown.max_drawdown)
    return (("Max drawdown (trades)", f"{format_money(amount)} ({share})"),)


def _curve_ratio_figures(metrics: EquityMetricsResult) -> tuple[tuple[str, str], ...]:
    return tuple(
        (label, format_ratio(value))
        for label, value in (
            ("Sharpe ratio", metrics.sharpe_ratio),
            ("Sortino ratio", metrics.sortino_ratio),
            ("Calmar ratio", metrics.calmar_ratio),
            ("Return over max drawdown", metrics.return_over_max_drawdown),
        )
        if value is not None
    )


def _trades_table(
    trade_metrics: TradeMetricsResult, sides: Sequence[TradeMetricsResult]
) -> ReportTable:
    return build_trades_table(
        [(_ALL_COLUMN, trade_metrics), *((_side_title(side), side) for side in sides)]
    )


def build_trades_table(
    measured: Sequence[tuple[str, TradeMetricsResult]], title: str = _TRADES_TITLE
) -> ReportTable:
    """Every figure here remains valid on a filtered subset of trades, which is
    what makes one row readable across its columns.

    Args:
        measured: The column title and the figures it shows, in column order.
    """
    columns = [column for column, _ in measured]
    figures = [metrics for _, metrics in measured]
    return ReportTable(
        title=title,
        columns=columns,
        groups=[
            _group(
                None,
                figures,
                ("Total trades", lambda m: str(m.total_trades)),
                ("Winning trades", lambda m: str(m.winning_trades)),
                ("Losing trades", lambda m: str(m.losing_trades)),
                ("Win rate", lambda m: format_percent(m.win_rate)),
            ),
            _group(
                "PnL",
                figures,
                ("Total PnL (closed)", lambda m: format_money(m.total_pnl)),
                ("Profit factor", lambda m: format_ratio(m.profit_factor)),
                ("Risk-reward ratio", lambda m: format_ratio(m.risk_reward_ratio)),
                ("Average PnL", lambda m: format_money(m.avg_trade_pnl)),
                (
                    "Average winning trade",
                    lambda m: format_money(m.avg_winning_trade_pnl),
                ),
                (
                    "Average losing trade",
                    lambda m: format_money(m.avg_losing_trade_pnl),
                ),
                (
                    "Largest winning trade",
                    lambda m: format_money(m.largest_winning_trade_pnl),
                ),
                (
                    "Largest losing trade",
                    lambda m: format_money(m.largest_losing_trade_pnl),
                ),
                ("Best trade return", lambda m: format_percent(m.best_trade_return)),
                ("Worst trade return", lambda m: format_percent(m.worst_trade_return)),
            ),
            _group(
                "Streaks & cadence",
                figures,
                ("Max win streak", lambda m: str(m.max_win_streak)),
                ("Max lose streak", lambda m: str(m.max_lose_streak)),
                (
                    "Average trades per day",
                    lambda m: format_ratio(m.avg_trades_per_day),
                ),
            ),
            _group(
                "Durations",
                figures,
                (
                    "Average trade duration",
                    lambda m: _days(m.avg_trade_duration_days),
                ),
                (
                    "Average winning trade duration",
                    lambda m: _days(m.avg_winning_trade_duration_days),
                ),
                (
                    "Average losing trade duration",
                    lambda m: _days(m.avg_losing_trade_duration_days),
                ),
            ),
            _group(
                "Fees",
                figures,
                ("Total fees", lambda m: format_money(m.total_fee)),
                ("Average fee", lambda m: format_money(m.avg_fee)),
            ),
        ],
    )


def _side_title(metrics: TradeMetricsResult) -> str:
    side = metrics.filter_applied.side
    return _SIDELESS_TITLE if side is None else side.capitalize()


def _group(
    title: str | None, measured: Sequence[TradeMetricsResult], *figures: _Figure
) -> ReportGroup:
    return ReportGroup(
        title=title,
        rows=[
            ReportTableRow(label=label, values=[value_of(one) for one in measured])
            for label, value_of in figures
        ],
    )


def _section(title: str, *figures: tuple[str, str]) -> ReportSection:
    return ReportSection(title=title, rows=_rows(*figures))


def _rows(*figures: tuple[str, str]) -> list[ReportRow]:
    return [ReportRow(label=label, value=value) for label, value in figures]


def _moment(moment: datetime) -> str:
    """Show a moment to the minute, which is as finely as a candle closes."""
    return moment.strftime("%Y-%m-%d %H:%M")


def _days(days: float) -> str:
    return f"{days:.2f} days"
