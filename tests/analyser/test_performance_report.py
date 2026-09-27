from collections.abc import Callable
from dataclasses import replace
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from robottraderslab._core import PerformanceReport
from robottraderslab.analyser import Analyser, TradeFilter
from robottraderslab.analyser.equity_metrics import (
    EquityMetricsResult,
    HodlComparison,
)
from robottraderslab.analyser.performance_report import build_performance_report
from robottraderslab.analyser.trade_metrics import (
    TradeDrawdownMetrics,
    TradeMetricsResult,
)


@pytest.fixture
def trade_metrics() -> TradeMetricsResult:
    return TradeMetricsResult(
        total_trades=4,
        winning_trades=3,
        losing_trades=1,
        win_rate=0.75,
        profit_factor=2.5,
        max_win_streak=2,
        max_lose_streak=1,
        total_fee=12.4,
        biggest_fee=5.0,
        avg_fee=3.1,
        avg_trade_pnl=42.5,
        avg_trade_return=1.5,
        avg_winning_trade_pnl=60.0,
        avg_losing_trade_pnl=20.0,
        avg_winning_trade_return=2.0,
        avg_losing_trade_return=1.0,
        largest_winning_trade_pnl=90.0,
        largest_losing_trade_pnl=20.0,
        best_trade_return=0.0325,
        worst_trade_return=-0.0175,
        total_pnl=170.0,
        avg_trade_duration_days=1.25,
        avg_winning_trade_duration_days=1.5,
        avg_losing_trade_duration_days=0.5,
        time_in_position_ratio=0.42,
        avg_trades_per_day=0.8,
        open_reasons=pd.Series(dtype=str),
        close_reasons=pd.Series(dtype=str),
        risk_reward_ratio=2.0,
        filter_applied=TradeFilter(),
    )


@pytest.fixture
def equity_metrics() -> EquityMetricsResult:
    return EquityMetricsResult(
        period_start=pd.Timestamp("2024-01-01"),
        period_end=pd.Timestamp("2024-01-02"),
        initial_equity=1000.0,
        final_equity=1450.0,
        net_profit=450.0,
        roi=0.45,
        max_drawdown=-0.115,
        max_drawdown_amount=-125.5,
        sharpe_ratio=1.234,
        sortino_ratio=1.8,
        calmar_ratio=3.9,
        drawdown_percentage=pd.Series(dtype=float),
        absolute_drawdown=pd.Series(dtype=float),
        returns=pd.Series(dtype=float),
        hodls=[HodlComparison(held="BTC", hodl_return=0.2, performance_vs_hodl=0.208)],
        return_over_max_drawdown=3.91,
        display_equity_curve=pd.Series(dtype=float),
    )


_TRADE_DRAWDOWN = TradeDrawdownMetrics(max_drawdown=-0.08, amount=-96.0)


@pytest.fixture
def trade_drawdown() -> TradeDrawdownMetrics:
    return _TRADE_DRAWDOWN


@pytest.fixture
def equity_curve() -> pd.Series:
    return pd.Series(
        [1000.0, 1450.0], index=pd.date_range("2024-01-01", periods=2, freq="D")
    )


@pytest.fixture
def report(
    trade_metrics: TradeMetricsResult, equity_metrics: EquityMetricsResult
) -> PerformanceReport:
    return build_performance_report(trade_metrics, equity_metrics, _TRADE_DRAWDOWN)


@pytest.fixture
def create_analyser(
    equity_metrics: EquityMetricsResult,
    trade_drawdown: TradeDrawdownMetrics,
    equity_curve: pd.Series,
    plotting_service: Mock,
    chart_service: Mock,
) -> Callable[..., Analyser]:
    def create(trade_metrics: TradeMetricsResult, **overrides: Any) -> Analyser:
        arguments: dict[str, Any] = {
            "trades": pd.DataFrame(),
            "open_positions": pd.DataFrame(),
            "equity_curve": equity_curve,
            "trade_metrics": trade_metrics,
            "equity_metrics": equity_metrics,
            "trade_drawdown": trade_drawdown,
            "plotting_service": plotting_service,
            "chart_service": chart_service,
            "profiles": [{"symbol": "BTC/USDT:USDT", "timeframe": "1h"}],
        }
        arguments.update(overrides)
        return Analyser(**arguments)

    return create


@pytest.fixture
def analyser(
    create_analyser: Callable[[TradeMetricsResult], Analyser],
    trade_metrics: TradeMetricsResult,
) -> Analyser:
    return create_analyser(trade_metrics)


def _draw(analyser):
    analyser.plot_candlestick(
        indicators_name="impulse",
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        indicators_params={},
    )


def _figure(report, label):
    return next(
        row.value
        for section in report.sections
        for row in section.rows
        if row.label == label
    )


def _cell(report, label, column):
    at = report.trades.columns.index(column)
    return next(
        row.values[at]
        for group in report.trades.groups
        for row in group.rows
        if row.label == label
    )


def _headline(report, label):
    return next(row.value for row in report.headline if row.label == label)


def _section_of(report, label):
    return next(
        section.title
        for section in report.sections
        for row in section.rows
        if row.label == label
    )


class TestPerformanceReport:
    @pytest.mark.parametrize(
        ("label", "shown"),
        [
            ("Hodl performance (BTC)", "20.00%"),
            ("Performance vs hodl (BTC)", "20.80%"),
            ("Time in position", "42.00%"),
        ],
    )
    def test_fractions_are_shown_as_percentages(self, report, label, shown):
        assert _figure(report, label) == shown

    @pytest.mark.parametrize(
        ("label", "shown"),
        [
            ("Sharpe ratio", "1.23"),
            ("Sortino ratio", "1.80"),
            ("Calmar ratio", "3.90"),
            ("Return over max drawdown", "3.91"),
        ],
    )
    def test_ratios_are_shown_unscaled(self, report, label, shown):
        assert _figure(report, label) == shown

    @pytest.mark.parametrize(
        ("label", "shown"),
        [
            ("Initial balance", "$1000.00"),
            ("Final equity", "$1450.00"),
        ],
    )
    def test_amounts_are_shown_as_money(self, report, label, shown):
        assert _figure(report, label) == shown

    @pytest.mark.parametrize(
        ("label", "shown"),
        [
            ("PnL", "$450.00 (45.00%)"),
            ("Max drawdown", "-$125.50 (-11.50%)"),
        ],
    )
    def test_a_figure_with_two_readings_shows_both(self, report, label, shown):
        assert _figure(report, label) == shown

    def test_the_period_is_shown_to_the_minute(self, report):
        assert _figure(report, "Period start") == "2024-01-01 00:00"
        assert _figure(report, "Period end") == "2024-01-02 00:00"

    @pytest.mark.parametrize(
        ("label", "shown"),
        [
            ("Profit factor", "2.50"),
            ("Sharpe ratio", "1.23"),
        ],
    )
    def test_the_headline_answers_whether_the_run_was_any_good(
        self, report, label, shown
    ):
        assert _headline(report, label) == shown

    def test_the_headline_drawdown_is_given_as_an_amount_and_a_share(self, report):
        assert _headline(report, "Max drawdown") == "-$125.50 (-11.50%)"

    def test_the_headline_pnl_is_given_as_an_amount_and_a_share(self, report):
        assert _headline(report, "PnL") == "$450.00 (45.00%)"

    def test_the_headline_win_rate_counts_the_trades_behind_it(self, report):
        assert _headline(report, "Win rate") == "75.00% (3/4)"

    def test_the_figures_are_grouped_into_titled_sections(self, report):
        titles = [section.title for section in report.sections]

        assert titles == ["Overview", "Risk & ratios"]

    @pytest.mark.parametrize(
        "label",
        ["Sharpe ratio", "Sortino ratio", "Calmar ratio", "Return over max drawdown"],
    )
    def test_risk_ratios_sit_with_the_other_health_measures(self, report, label):
        assert _section_of(report, label) == "Risk & ratios"


class TestTradesTable:
    @pytest.mark.parametrize(
        ("label", "shown"),
        [
            ("Total trades", "4"),
            ("Winning trades", "3"),
            ("Losing trades", "1"),
            ("Max win streak", "2"),
            ("Max lose streak", "1"),
        ],
    )
    def test_counts_are_shown_whole(self, report, label, shown):
        assert _cell(report, label, "All") == shown

    @pytest.mark.parametrize(
        ("label", "shown"),
        [
            ("Total PnL (closed)", "$170.00"),
            ("Average PnL", "$42.50"),
            ("Average winning trade", "$60.00"),
            ("Average losing trade", "$20.00"),
            ("Largest winning trade", "$90.00"),
            ("Largest losing trade", "$20.00"),
            ("Total fees", "$12.40"),
            ("Average fee", "$3.10"),
        ],
    )
    def test_amounts_are_shown_as_money(self, report, label, shown):
        assert _cell(report, label, "All") == shown

    @pytest.mark.parametrize(
        ("label", "shown"),
        [
            ("Win rate", "75.00%"),
            ("Profit factor", "2.50"),
            ("Risk-reward ratio", "2.00"),
            ("Average trades per day", "0.80"),
        ],
    )
    def test_rates_and_ratios_are_shown_per_column(self, report, label, shown):
        assert _cell(report, label, "All") == shown

    @pytest.mark.parametrize(
        ("label", "shown"),
        [
            ("Average trade duration", "1.25 days"),
            ("Average winning trade duration", "1.50 days"),
            ("Average losing trade duration", "0.50 days"),
        ],
    )
    def test_durations_name_their_unit(self, report, label, shown):
        assert _cell(report, label, "All") == shown

    def test_the_rows_are_grouped_by_what_they_measure(self, report):
        titles = [group.title for group in report.trades.groups]

        assert titles == [None, "PnL", "Streaks & cadence", "Durations", "Fees"]

    def test_a_run_without_sides_measures_one_column(self, report):
        assert report.trades.columns == ["All"]


class TestTheTradesDrawdown:
    def test_the_fall_is_shown_as_an_amount_and_a_share(self, report):
        assert _figure(report, "Max drawdown (trades)") == "-$96.00 (-8.00%)"

    def test_a_report_measured_from_no_balance_states_no_share(
        self, trade_metrics, equity_metrics
    ):
        unmeasured = replace(equity_metrics, initial_equity=None, final_equity=None)

        report = build_performance_report(trade_metrics, unmeasured, _TRADE_DRAWDOWN)

        labels = {row.label for section in report.sections for row in section.rows}
        assert "Max drawdown (trades)" not in labels

    def test_a_window_with_no_peak_to_fall_from_states_no_share(
        self, trade_metrics, equity_metrics
    ):
        untraded = replace(_TRADE_DRAWDOWN, amount=None)

        report = build_performance_report(trade_metrics, equity_metrics, untraded)

        labels = {row.label for section in report.sections for row in section.rows}
        assert "Max drawdown (trades)" not in labels


class TestRunWithoutAReferencePrice:
    def test_the_hodl_figures_are_absent_rather_than_zero(
        self, trade_metrics, equity_metrics
    ):
        unreferenced = replace(equity_metrics, hodls=[])

        report = build_performance_report(trade_metrics, unreferenced, _TRADE_DRAWDOWN)

        labels = {row.label for section in report.sections for row in section.rows}
        assert "Hodl performance" not in labels
        assert "Performance vs hodl" not in labels


class TestRunMeasuredAgainstTwoBenchmarks:
    def test_each_benchmark_names_its_own_rows(self, trade_metrics, equity_metrics):
        both = replace(
            equity_metrics,
            hodls=[
                HodlComparison(held="BTC", hodl_return=0.2, performance_vs_hodl=0.208),
                HodlComparison(held="basket", hodl_return=0.1, performance_vs_hodl=0.4),
            ],
        )

        report = build_performance_report(trade_metrics, both, _TRADE_DRAWDOWN)

        assert _figure(report, "Hodl performance (BTC)") == "20.00%"
        assert _figure(report, "Hodl performance (basket)") == "10.00%"
        assert _figure(report, "Performance vs hodl (basket)") == "40.00%"


class TestStrategyWithoutALosingTrade:
    def test_a_profit_factor_with_nothing_to_divide_by(
        self, trade_metrics, equity_metrics
    ):
        unbeaten = replace(trade_metrics, profit_factor=float("inf"))

        report = build_performance_report(unbeaten, equity_metrics, _TRADE_DRAWDOWN)

        assert _cell(report, "Profit factor", "All") == "∞"


class TestLosingStrategy:
    def test_an_amount_below_zero(self, trade_metrics, equity_metrics):
        losing = replace(trade_metrics, avg_trade_pnl=-18.29)

        report = build_performance_report(losing, equity_metrics, _TRADE_DRAWDOWN)

        assert _cell(report, "Average PnL", "All") == "-$18.29"


class TestPerformanceReportOutputs:
    def test_the_printed_summary_titles_each_section(self, analyser, capsys):
        analyser.print_performance_summary()

        assert "--- Risk & ratios ---" in capsys.readouterr().out

    def test_the_printed_summary_shows_a_figure_beside_its_label(
        self, analyser, capsys
    ):
        analyser.print_performance_summary()

        assert "Sharpe ratio: 1.23" in capsys.readouterr().out

    def test_the_printed_table_names_its_columns(self, analyser, capsys):
        analyser.print_performance_summary()

        printed = capsys.readouterr().out
        assert "--- Trades ---" in printed
        assert " " * 32 + "         All" in printed

    def test_the_printed_table_aligns_a_figure_under_its_column(self, analyser, capsys):
        analyser.print_performance_summary()

        assert "Total trades" + " " * 20 + " " * 11 + "4" in capsys.readouterr().out

    def test_the_printed_table_titles_its_groups(self, analyser, capsys):
        analyser.print_performance_summary()

        assert "\nStreaks & cadence\n" in capsys.readouterr().out

    def test_the_chart_is_handed_the_same_report(self, analyser, chart_service, report):
        _draw(analyser)

        assert chart_service.open_candlestick.call_args.kwargs["report"] == report


class TestMultiChartRun:
    def test_every_chart_of_the_run_is_handed_the_report(
        self, analyser, chart_service, report
    ):
        analyser.plot_candlesticks(indicators_name="impulse")

        assert chart_service.open_candlesticks.call_args.kwargs["report"] == report


@pytest.fixture
def two_sided_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "side": ["long", "long", "short"],
            "net_pnl": [100.0, -40.0, 60.0],
            "net_pnl_pct": [0.10, -0.04, 0.06],
            "fee": [1.0, 1.0, 2.0],
            "entry_fee": [0.5, 0.5, 1.0],
            "exit_fee": [0.5, 0.5, 1.0],
            "entry_time": pd.to_datetime(
                ["2024-01-01 10:00", "2024-01-02 10:00", "2024-01-03 10:00"]
            ),
            "exit_time": pd.to_datetime(
                ["2024-01-01 18:00", "2024-01-02 18:00", "2024-01-03 18:00"]
            ),
            "entry_reason": ["signal", "signal", "signal"],
            "exit_reason": ["exit", "exit", "exit"],
            "tag": ["a", "a", "a"],
        }
    )


class TestTradeReturns:
    def test_the_best_and_worst_trade_returns_are_shown_as_percentages(self, report):
        assert _cell(report, "Best trade return", "All") == "3.25%"
        assert _cell(report, "Worst trade return", "All") == "-1.75%"


class TestSidesOfTheRun:
    def test_each_side_that_traded_gets_a_column(
        self, create_analyser, trade_metrics, two_sided_trades
    ):
        report = create_analyser(
            trade_metrics, trades=two_sided_trades
        ).performance_report()

        assert report.trades.columns == ["All", "Long", "Short"]

    def test_a_side_never_taken_is_absent(
        self, create_analyser, trade_metrics, two_sided_trades
    ):
        long_only = two_sided_trades[two_sided_trades["side"] == "long"]

        report = create_analyser(trade_metrics, trades=long_only).performance_report()

        assert report.trades.columns == ["All", "Long"]

    def test_a_side_carries_its_own_trade_figures(
        self, create_analyser, trade_metrics, two_sided_trades
    ):
        report = create_analyser(
            trade_metrics, trades=two_sided_trades
        ).performance_report()

        assert _cell(report, "Total trades", "All") == "4"
        assert _cell(report, "Total trades", "Long") == "2"
        assert _cell(report, "Total trades", "Short") == "1"

    def test_a_run_with_no_trades_has_no_side_columns(self, analyser):
        assert analyser.performance_report().trades.columns == ["All"]


@pytest.fixture
def unmeasured_metrics(equity_metrics: EquityMetricsResult) -> EquityMetricsResult:
    """The same run with nothing stating what its curve is measured from."""
    return replace(
        equity_metrics,
        initial_equity=None,
        final_equity=None,
        roi=None,
        max_drawdown=None,
        sharpe_ratio=None,
        sortino_ratio=None,
        calmar_ratio=None,
        return_over_max_drawdown=None,
        drawdown_percentage=None,
        returns=None,
        hodls=[],
    )


def _labels(report):
    return [row.label for section in report.sections for row in section.rows] + [
        row.label for row in report.headline
    ]


class TestReportWithoutABalance:
    def test_the_profit_is_shown_as_an_amount_alone(
        self, trade_metrics, unmeasured_metrics
    ):
        report = build_performance_report(
            trade_metrics, unmeasured_metrics, _TRADE_DRAWDOWN
        )

        assert _headline(report, "PnL") == "$450.00"

    def test_the_drawdown_is_shown_as_an_amount_alone(
        self, trade_metrics, unmeasured_metrics
    ):
        report = build_performance_report(
            trade_metrics, unmeasured_metrics, _TRADE_DRAWDOWN
        )

        assert _headline(report, "Max drawdown") == "-$125.50"

    @pytest.mark.parametrize(
        "label",
        [
            "Initial balance",
            "Final equity",
            "Sharpe ratio",
            "Sortino ratio",
            "Calmar ratio",
            "Return over max drawdown",
        ],
    )
    def test_a_figure_it_cannot_state_is_left_out(
        self, trade_metrics, unmeasured_metrics, label
    ):
        report = build_performance_report(
            trade_metrics, unmeasured_metrics, _TRADE_DRAWDOWN
        )

        assert label not in _labels(report)

    @pytest.mark.parametrize(
        "label", ["PnL", "Max drawdown", "Win rate", "Profit factor"]
    )
    def test_a_figure_it_can_state_is_kept(
        self, trade_metrics, unmeasured_metrics, label
    ):
        report = build_performance_report(
            trade_metrics, unmeasured_metrics, _TRADE_DRAWDOWN
        )

        assert label in _labels(report)

    def test_the_trades_table_reads_the_same_either_way(
        self, trade_metrics, equity_metrics, unmeasured_metrics
    ):
        measured = build_performance_report(
            trade_metrics, equity_metrics, _TRADE_DRAWDOWN
        )
        unmeasured = build_performance_report(
            trade_metrics, unmeasured_metrics, _TRADE_DRAWDOWN
        )

        assert _cell(unmeasured, "Total trades", "All") == _cell(
            measured, "Total trades", "All"
        )
