import re
from collections.abc import Callable
from unittest.mock import Mock

import pandas as pd
import pytest

from robottraderslab.analyser import Analyser, AnalysisInputs, TradeFilter
from robottraderslab.bootstrap import ReportConfig


@pytest.fixture
def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [
                "BTC/USDT:USDT",
                "BTC/USDT:USDT",
                "ETH/USDT:USDT",
                "ETH/USDT:USDT",
                "BTC/USDT:USDT",
                "ETH/USDT:USDT",
            ],
            "side": ["long", "short", "long", "short", "long", "long"],
            "entry_time": pd.to_datetime(
                [
                    "2024-01-01 10:00",
                    "2024-01-02 11:00",
                    "2024-01-03 12:00",
                    "2024-01-04 13:00",
                    "2024-01-05 14:00",
                    "2024-01-06 15:00",
                ]
            ),
            "exit_time": pd.to_datetime(
                [
                    "2024-01-01 18:00",
                    "2024-01-02 19:00",
                    "2024-01-03 20:00",
                    "2024-01-04 21:00",
                    "2024-01-05 22:00",
                    "2024-01-06 23:00",
                ]
            ),
            "net_pnl": [100.0, -50.0, 75.0, -25.0, 150.0, 80.0],
            "net_pnl_pct": [0.02, -0.01, 0.015, -0.005, 0.03, 0.016],
            "entry_fee": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "exit_fee": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "entry_reason": [
                "signal_a",
                "signal_b",
                "signal_a",
                "signal_b",
                "signal_a",
                "signal_a",
            ],
            "exit_reason": ["tp", "sl", "tp", "sl", "tp", "tp"],
        }
    )


@pytest.fixture
def equity_curve() -> pd.Series:
    return pd.Series(
        [10000, 10100, 10050, 10125, 10100, 10250, 10330],
        index=pd.date_range("2024-01-01", periods=7, freq="D"),
    )


@pytest.fixture
def create_analyser(
    equity_curve: pd.Series, ohlcv_provider: Mock
) -> Callable[..., Analyser]:
    def create(trades: pd.DataFrame) -> Analyser:
        inputs = AnalysisInputs(
            trades=trades,
            open_positions=pd.DataFrame(),
            equity_curve=equity_curve,
            initial_balance=float(equity_curve.iloc[0]),
        )
        return Analyser.from_analysis_inputs(inputs, ohlcv_provider, ReportConfig())

    return create


@pytest.fixture
def analyser(
    create_analyser: Callable[..., Analyser], trades: pd.DataFrame
) -> Analyser:
    return create_analyser(trades)


class TestLongShortAnalysis:
    def test_the_sides_are_compared_side_by_side(self, analyser, capsys):
        analyser.print_long_short_analysis()

        printed = capsys.readouterr().out
        assert "LONG vs SHORT TRADE ANALYSIS" in printed
        assert re.search(r"Total trades\s+4\s+2\b", printed)

    def test_a_run_without_short_trades_cannot_be_compared(
        self, create_analyser, trades
    ):
        long_only = create_analyser(trades[trades["side"] == "long"])

        with pytest.raises(ValueError, match="No short trades found"):
            long_only.print_long_short_analysis()

    def test_the_averages_are_stated_as_shares_of_the_initial_equity(
        self, analyser, capsys
    ):
        analyser.print_long_short_analysis()

        printed = capsys.readouterr().out
        shares = printed.split("As a share of the initial equity\n")[1]
        assert re.match(r"Average PnL\s+1\.01%\s+-0\.38%", shares)


class TestSymbolAnalysis:
    def test_a_symbol_is_measured_on_its_own_trades(self, analyser, capsys):
        analyser.print_symbol_analysis("BTC/USDT:USDT")

        printed = capsys.readouterr().out
        assert "SYMBOL ANALYSIS: BTC/USDT:USDT" in printed
        assert re.search(r"Total trades\s+3\b", printed)

    def test_a_symbol_the_run_never_traded_has_nothing_to_measure(self, analyser):
        with pytest.raises(
            ValueError, match="No trades found matching filter criteria"
        ):
            analyser.print_symbol_analysis("NONEXISTENT/USDT:USDT")


class TestFilteredMetrics:
    def test_the_criteria_title_the_analysis(self, analyser, capsys):
        analyser.print_filtered_metrics(TradeFilter(side="long"))

        printed = capsys.readouterr().out
        assert "FILTERED ANALYSIS: long" in printed
        assert re.search(r"Total trades\s+4\b", printed)
