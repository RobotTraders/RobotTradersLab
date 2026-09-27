from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from robottraderslab._core import DrawdownUnit
from robottraderslab.analyser import Analyser, AnalysisInputs
from robottraderslab.analyser.equity_metrics import compute_equity_metrics
from robottraderslab.analyser.trade_metrics import (
    calculate_trade_based_max_drawdown,
    compute_trade_metrics,
)
from robottraderslab.bootstrap import ReportConfig

_REFERENCE_SYMBOL = "BTC/USDT:USDT"


@pytest.fixture
def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "net_pnl": [100.0, -50.0, 150.0, -30.0],
            "entry_fee": [1.0, 2.0, 1.5, 1.8],
            "exit_fee": [1.0, 2.0, 1.5, 1.8],
            "entry_time": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-02 14:00:00",
                    "2024-01-03 09:00:00",
                    "2024-01-04 16:00:00",
                ]
            ),
            "exit_time": pd.to_datetime(
                [
                    "2024-01-01 18:00:00",
                    "2024-01-04 14:00:00",
                    "2024-01-05 09:00:00",
                    "2024-01-06 16:00:00",
                ]
            ),
            "entry_reason": ["signal", "signal", "manual", "signal"],
            "exit_reason": ["take_profit", "stop_loss", "take_profit", "signal"],
            "net_pnl_pct": [0.01, -0.005, 0.015, -0.003],
        }
    )


@pytest.fixture
def open_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "side": ["long", "short"],
            "size": [1.0, 2.0],
            "entry_price": [50000.0, 3000.0],
            "unrealized_pnl": [100.0, -50.0],
        }
    )


@pytest.fixture
def equity_curve() -> pd.Series:
    return pd.Series(
        [1000, 1100, 1050, 1200, 1150, 1300, 1250, 1150, 1100, 1450],
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
        name="equity",
    )


@pytest.fixture
def reference_price(equity_curve) -> pd.Series:
    return pd.Series(
        np.linspace(100.0, 120.0, len(equity_curve)), index=equity_curve.index
    )


@pytest.fixture
def inputs(trades, open_positions, equity_curve) -> AnalysisInputs:
    return AnalysisInputs(
        trades=trades,
        open_positions=open_positions,
        equity_curve=equity_curve,
        initial_balance=float(equity_curve.iloc[0]),
    )


@pytest.fixture
def inputs_with_reference(inputs, reference_price) -> AnalysisInputs:
    return AnalysisInputs(
        trades=inputs.trades,
        open_positions=inputs.open_positions,
        equity_curve=inputs.equity_curve,
        initial_balance=inputs.initial_balance,
        reference_price=reference_price,
        reference_symbol_str=_REFERENCE_SYMBOL,
    )


@pytest.fixture
def inputs_without_balance(inputs) -> AnalysisInputs:
    return AnalysisInputs(
        trades=inputs.trades,
        open_positions=inputs.open_positions,
        equity_curve=inputs.equity_curve,
    )


@pytest.fixture
def create_analyser(
    ohlcv_provider, plotting_service, chart_service
) -> Callable[..., Analyser]:
    """Measures the inputs the way the engine does and fakes what draws them."""

    def create(inputs: AnalysisInputs, **overrides: Any) -> Analyser:
        equity_metrics = compute_equity_metrics(
            inputs.equity_curve,
            inputs.trades,
            ohlcv_provider,
            price_series=inputs.reference_price,
            reference_name=inputs.reference_symbol_str,
            initial_balance=inputs.initial_balance,
        )
        return Analyser(
            trades=inputs.trades,
            open_positions=inputs.open_positions,
            equity_curve=inputs.equity_curve,
            trade_metrics=compute_trade_metrics(inputs.trades),
            equity_metrics=equity_metrics,
            trade_drawdown=calculate_trade_based_max_drawdown(
                inputs.equity_curve, inputs.trades
            ),
            plotting_service=plotting_service,
            chart_service=chart_service,
            reference_price=inputs.reference_price,
            reference_symbol_str=inputs.reference_symbol_str,
            **overrides,
        )

    return create


@pytest.fixture
def analyser(create_analyser, inputs) -> Analyser:
    return create_analyser(inputs)


@pytest.fixture
def analyser_with_reference(create_analyser, inputs_with_reference) -> Analyser:
    return create_analyser(inputs_with_reference)


@pytest.fixture
def analyser_without_balance(create_analyser, inputs_without_balance) -> Analyser:
    return create_analyser(inputs_without_balance)


class TestInputs:
    @pytest.mark.parametrize("handed_out", ["trades", "open_positions", "equity_curve"])
    def test_an_input_is_handed_out_as_a_copy(self, analyser, inputs, handed_out):
        assert getattr(analyser, handed_out) is not getattr(inputs, handed_out)

    def test_a_run_without_trades_counts_none(
        self, create_analyser, trades, open_positions, equity_curve
    ):
        no_trades = AnalysisInputs(
            trades=trades.iloc[0:0],
            open_positions=open_positions,
            equity_curve=equity_curve,
            initial_balance=float(equity_curve.iloc[0]),
        )

        assert create_analyser(no_trades).closed_trades_count == 0


class TestTradeFigures:
    @pytest.mark.parametrize(
        ("figure", "expected"),
        [
            ("closed_trades_count", 4),
            ("winning_trades_count", 2),
            ("losing_trades_count", 2),
            ("win_rate", 0.5),
            ("risk_reward_ratio", 250.0 / 80.0),
            ("profit_factor", 250.0 / 80.0),
            ("total_pnl", 170.0),
            ("avg_trade_pnl", 42.5),
            ("avg_winning_trade_pnl", 125.0),
            ("avg_losing_trade_pnl", 40.0),
            ("largest_winning_trade_pnl", 150.0),
            ("largest_losing_trade_pnl", 50.0),
            ("total_fees", 12.6),
        ],
    )
    def test_a_figure_is_measured_over_the_trades(self, analyser, figure, expected):
        assert getattr(analyser, figure) == pytest.approx(expected)

    def test_the_entry_reasons_are_counted(self, analyser):
        assert analyser.open_reasons.to_dict() == {"signal": 3, "manual": 1}

    def test_the_exit_reasons_are_counted(self, analyser):
        assert analyser.close_reasons.to_dict() == {
            "take_profit": 2,
            "stop_loss": 1,
            "signal": 1,
        }


class TestEquityFigures:
    def test_the_return_on_the_opening_balance(self, analyser):
        assert analyser.roi == pytest.approx(0.45)

    def test_the_deepest_fall_from_a_peak(self, analyser):
        assert analyser.max_drawdown == pytest.approx(-200.0 / 1300.0)

    @pytest.mark.parametrize(
        "series",
        [
            "drawdown_series",
            "absolute_drawdown_series",
            "returns",
            "open_reasons",
            "close_reasons",
        ],
    )
    def test_a_series_is_handed_out_as_a_copy(self, analyser, series):
        assert getattr(analyser, series) is not getattr(analyser, series)


class TestHoldingTheReference:
    def test_what_holding_the_reference_would_have_returned(
        self, analyser_with_reference
    ):
        assert analyser_with_reference.hodl_return == pytest.approx(0.2)

    def test_the_run_measured_against_holding_the_reference(
        self, analyser_with_reference
    ):
        assert analyser_with_reference.performance_vs_hodl == pytest.approx(
            1.45 / 1.2 - 1.0
        )

    def test_without_a_reference_there_is_nothing_to_hold(self, analyser):
        assert analyser.hodl_return is None

    def test_without_a_reference_the_run_stands_uncompared(self, analyser):
        assert analyser.performance_vs_hodl is None


class TestPlotting:
    def test_the_equity_curve_is_drawn_with_its_figures(
        self, analyser, plotting_service, equity_curve
    ):
        analyser.plot_equity_curve(title="Test", filename="test")

        drawn = plotting_service.plot_equity_curve.call_args.kwargs
        assert drawn["title"] == "Test"
        assert drawn["filename"] == "test"
        pd.testing.assert_series_equal(drawn["equity_curve"], equity_curve)
        assert drawn["roi"] == pytest.approx(0.45)
        assert drawn["max_drawdown"] == pytest.approx(-200.0 / 1300.0)

    def test_the_reference_price_is_drawn_over_the_equity(
        self, analyser_with_reference, plotting_service, reference_price
    ):
        analyser_with_reference.plot_equity_curve()

        drawn = plotting_service.plot_equity_curve.call_args.kwargs
        pd.testing.assert_series_equal(drawn["price_series"], reference_price)
        assert drawn["price_label"] == _REFERENCE_SYMBOL

    def test_the_reference_price_can_be_left_out(
        self, analyser_with_reference, plotting_service
    ):
        analyser_with_reference.plot_equity_curve(plot_price=False)

        assert (
            plotting_service.plot_equity_curve.call_args.kwargs["price_series"] is None
        )

    def test_the_drawdown_is_drawn_as_a_percentage_of_the_peak(
        self, analyser, plotting_service, equity_curve
    ):
        analyser.plot_drawdown()

        drawn = plotting_service.plot_drawdown.call_args.kwargs
        assert drawn["unit"] is DrawdownUnit.PERCENT
        pd.testing.assert_series_equal(
            drawn["drawdown"], _percent_drawdown(equity_curve), check_names=False
        )

    def test_without_a_balance_the_drawdown_is_drawn_in_currency(
        self, analyser_without_balance, plotting_service, equity_curve
    ):
        analyser_without_balance.plot_drawdown()

        drawn = plotting_service.plot_drawdown.call_args.kwargs
        assert drawn["unit"] is DrawdownUnit.CURRENCY
        pd.testing.assert_series_equal(
            drawn["drawdown"], _currency_drawdown(equity_curve), check_names=False
        )

    def test_the_cumulative_pnl_is_a_share_of_the_opening_balance(
        self, analyser, plotting_service
    ):
        analyser.plot_cumulative_pnl_by_trade()

        drawn = plotting_service.plot_cumulative_pnl_by_trade.call_args
        pd.testing.assert_series_equal(
            drawn.args[0], pd.Series([10.0, 5.0, 20.0, 17.0]), check_names=False
        )
        assert drawn.kwargs["show_percentage"] is True

    def test_the_cumulative_pnl_can_be_drawn_in_currency(
        self, analyser, plotting_service
    ):
        analyser.plot_cumulative_pnl_by_trade(show_percentage=False)

        drawn = plotting_service.plot_cumulative_pnl_by_trade.call_args
        pd.testing.assert_series_equal(
            drawn.args[0], pd.Series([100.0, 50.0, 200.0, 170.0]), check_names=False
        )
        assert drawn.kwargs["show_percentage"] is False

    def test_the_cumulative_pnl_is_drawn_oldest_trade_first(
        self, create_analyser, open_positions, equity_curve, plotting_service
    ):
        newest_first = AnalysisInputs(
            trades=pd.DataFrame(
                {
                    "net_pnl": [150.0, -50.0, 100.0],
                    "entry_fee": [0.0, 0.0, 0.0],
                    "exit_fee": [0.0, 0.0, 0.0],
                    "entry_time": pd.to_datetime(
                        ["2024-01-03", "2024-01-02", "2024-01-01"]
                    ),
                    "exit_time": pd.to_datetime(
                        ["2024-01-03", "2024-01-02", "2024-01-01"]
                    ),
                    "entry_reason": ["signal", "signal", "signal"],
                    "exit_reason": ["tp", "sl", "tp"],
                    "net_pnl_pct": [0.015, -0.005, 0.01],
                }
            ),
            open_positions=open_positions,
            equity_curve=equity_curve,
            initial_balance=float(equity_curve.iloc[0]),
        )

        create_analyser(newest_first).plot_cumulative_pnl_by_trade(
            show_percentage=False
        )

        drawn = plotting_service.plot_cumulative_pnl_by_trade.call_args.args[0]
        pd.testing.assert_series_equal(
            drawn, pd.Series([100.0, 50.0, 200.0]), check_names=False
        )

    def test_a_liquidation_does_not_zero_the_cumulative_pnl(
        self, create_analyser, open_positions, equity_curve, plotting_service
    ):
        liquidated = AnalysisInputs(
            trades=pd.DataFrame(
                {
                    "net_pnl": [10.0, -100.0, 20.0],
                    "entry_fee": [0.0, 0.0, 0.0],
                    "exit_fee": [0.0, 0.0, 0.0],
                    "entry_time": pd.to_datetime(
                        ["2024-01-01", "2024-01-02", "2024-01-03"]
                    ),
                    "exit_time": pd.to_datetime(
                        ["2024-01-01", "2024-01-02", "2024-01-03"]
                    ),
                    "entry_reason": ["signal", "signal", "signal"],
                    "exit_reason": ["tp", "liquidation", "tp"],
                    "net_pnl_pct": [0.1, -1.0, 0.2],
                }
            ),
            open_positions=open_positions,
            equity_curve=equity_curve,
            initial_balance=float(equity_curve.iloc[0]),
        )

        create_analyser(liquidated).plot_cumulative_pnl_by_trade()

        drawn = plotting_service.plot_cumulative_pnl_by_trade.call_args.args[0]
        assert drawn.iloc[-1] == pytest.approx(-7.0)

    def test_the_summary_is_drawn_from_the_report_and_the_curves(
        self, analyser_with_reference, plotting_service, equity_curve, reference_price
    ):
        analyser_with_reference.plot_performance_summary()

        drawn = plotting_service.plot_performance_summary.call_args.kwargs
        assert drawn["report"] == analyser_with_reference.performance_report()
        assert drawn["drawdown_unit"] is DrawdownUnit.PERCENT
        pd.testing.assert_series_equal(
            drawn["drawdown"], _percent_drawdown(equity_curve), check_names=False
        )
        pd.testing.assert_series_equal(drawn["price_series"], reference_price)
        assert drawn["price_label"] == _REFERENCE_SYMBOL

    def test_without_a_balance_the_summary_drawdown_is_in_currency(
        self, analyser_without_balance, plotting_service, equity_curve
    ):
        analyser_without_balance.plot_performance_summary()

        drawn = plotting_service.plot_performance_summary.call_args.kwargs
        assert drawn["drawdown_unit"] is DrawdownUnit.CURRENCY
        pd.testing.assert_series_equal(
            drawn["drawdown"], _currency_drawdown(equity_curve), check_names=False
        )

    def test_a_year_of_monthly_performance(
        self, analyser, plotting_service, equity_curve
    ):
        analyser.plot_monthly_performance(year=2024)

        drawn = plotting_service.plot_monthly_performance.call_args.kwargs
        assert drawn["year"] == 2024
        assert drawn["title"] is None
        assert drawn["filename"] is None
        pd.testing.assert_series_equal(drawn["equity_curve"], equity_curve)

    def test_the_last_year_of_the_run_is_drawn_by_default(
        self, analyser, plotting_service
    ):
        analyser.plot_monthly_performance()

        assert (
            plotting_service.plot_monthly_performance.call_args.kwargs["year"] == 2024
        )

    def test_every_year_of_the_run_gets_its_own_figure(
        self, create_analyser, trades, open_positions, plotting_service
    ):
        two_years = AnalysisInputs(
            trades=trades,
            open_positions=open_positions,
            equity_curve=pd.Series(
                range(1000, 1007),
                index=pd.date_range("2023-12-30", periods=7, freq="D"),
            ),
            initial_balance=1000.0,
        )

        create_analyser(two_years).plot_monthly_performance(
            year="all", filename="monthly"
        )

        drawn = plotting_service.plot_monthly_performance.call_args_list
        assert [call.kwargs["year"] for call in drawn] == [2023, 2024]
        assert [call.kwargs["filename"] for call in drawn] == [
            "monthly_2023",
            "monthly_2024",
        ]


class TestCharts:
    @pytest.fixture
    def profiles(self) -> list[dict[str, Any]]:
        return [
            {"symbol": "BTC/USDT:USDT", "timeframe": "1h", "tag": "alpha"},
            {"symbol": "ETH/USDT:USDT", "timeframe": "4h", "tag": "beta"},
        ]

    @pytest.fixture
    def tagged_inputs(self, inputs) -> AnalysisInputs:
        return AnalysisInputs(
            trades=inputs.trades.assign(
                symbol=["BTC/USDT:USDT", "ETH/USDT:USDT"] * 2,
                tag=["alpha", "beta"] * 2,
            ),
            open_positions=inputs.open_positions,
            equity_curve=inputs.equity_curve,
            initial_balance=inputs.initial_balance,
        )

    def test_the_declared_profiles_are_the_charts_drawn(
        self, create_analyser, tagged_inputs, profiles, chart_service
    ):
        create_analyser(tagged_inputs, profiles=profiles).plot_candlesticks(
            indicators_name="impulse"
        )

        assert chart_service.open_candlesticks.call_args.kwargs["charts"] == profiles

    def test_the_declared_profiles_name_the_trades_they_made(
        self, tagged_inputs, profiles, ohlcv_provider
    ):
        analyser = Analyser.from_analysis_inputs(
            tagged_inputs, ohlcv_provider, ReportConfig(), profiles=profiles
        )

        assert list(analyser.trades["profile_name"]) == [
            "BTC/USDT:USDT@1h-alpha",
            "ETH/USDT:USDT@4h-beta",
            "BTC/USDT:USDT@1h-alpha",
            "ETH/USDT:USDT@4h-beta",
        ]

    def test_a_run_declaring_no_profiles_has_nothing_to_chart(self, analyser):
        with pytest.raises(RuntimeError, match="No strategy profiles"):
            analyser.plot_candlesticks(indicators_name="impulse")


class TestPrintedSummary:
    def test_the_summary_is_printed_section_by_section(self, analyser, capsys):
        analyser.print_performance_summary()

        printed = capsys.readouterr().out
        assert "--- Overview ---" in printed
        assert "--- Risk & ratios ---" in printed
        assert "--- Trades ---" in printed
        assert "Largest winning trade" in printed

    def test_the_reason_tables_follow_the_summary(self, analyser, capsys):
        analyser.print_performance_summary()

        printed = capsys.readouterr().out
        assert "--- Entry Reasons ---" in printed
        assert "--- Exit Reasons ---" in printed

    def test_the_reason_pairs_name_their_key_without_a_row_number(
        self, analyser, capsys
    ):
        analyser.print_performance_summary()

        printed = capsys.readouterr().out
        pairs = printed.split("--- Entry to Exit Reason Pairs ---\n")[1]
        assert pairs.startswith("Entry Reason -> Exit Reason")


def _percent_drawdown(equity_curve: pd.Series) -> pd.Series:
    peak = equity_curve.cummax()
    return (equity_curve - peak) / peak * 100.0


def _currency_drawdown(equity_curve: pd.Series) -> pd.Series:
    return equity_curve - equity_curve.cummax()
