import pandas as pd
import pytest

from robottraderslab.analyser.trade_metrics.reason_analysis import (
    calculate_entry_exit_pairs_table,
    calculate_reason_summary_table,
)

SUMMARY_COLUMNS = ["Count", "% of Total", "Win Rate", "Avg PnL"]
PAIRS_COLUMNS = [
    "Entry Reason -> Exit Reason",
    "Count",
    "% of Total",
    "Win Rate",
    "Avg PnL",
]


@pytest.fixture
def trades_with_reasons() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_reason": ["signal", "signal", "manual"],
            "exit_reason": ["take_profit", "stop_loss", "take_profit"],
            "net_pnl": [100.0, -50.0, 75.0],
            "net_pnl_pct": [0.10, -0.05, 0.075],
        }
    )


@pytest.fixture
def trades_with_all_nan_reasons() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_reason": [None, None, None],
            "exit_reason": [None, None, None],
            "net_pnl": [100.0, -50.0, 75.0],
            "net_pnl_pct": [0.10, -0.05, 0.075],
        }
    )


@pytest.fixture
def trades_with_partial_reasons() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_reason": ["signal", None, "manual"],
            "exit_reason": [None, "stop_loss", "take_profit"],
            "net_pnl": [100.0, -50.0, 75.0],
            "net_pnl_pct": [0.10, -0.05, 0.075],
        }
    )


class TestCalculateReasonSummaryTable:
    def test_with_reasons(self, trades_with_reasons):
        summary = calculate_reason_summary_table(trades_with_reasons, "entry_reason")

        assert list(summary.columns) == SUMMARY_COLUMNS
        assert summary.index.name == "Reason"
        assert summary.loc["signal", "Count"] == 2
        assert summary.loc["manual", "Count"] == 1

    def test_all_nan_reasons(self, trades_with_all_nan_reasons):
        summary = calculate_reason_summary_table(
            trades_with_all_nan_reasons, "entry_reason"
        )

        assert summary.empty
        assert list(summary.columns) == SUMMARY_COLUMNS
        assert summary.index.name == "Reason"

    def test_partial_reasons(self, trades_with_partial_reasons):
        summary = calculate_reason_summary_table(
            trades_with_partial_reasons, "entry_reason"
        )

        assert len(summary) == 2
        assert summary.loc["signal", "Count"] == 1
        assert summary.loc["manual", "Count"] == 1


class TestCalculateEntryExitPairsTable:
    def test_with_reasons(self, trades_with_reasons):
        pairs = calculate_entry_exit_pairs_table(trades_with_reasons)

        assert list(pairs.columns) == PAIRS_COLUMNS
        assert len(pairs) == 3

    def test_all_nan_reasons(self, trades_with_all_nan_reasons):
        pairs = calculate_entry_exit_pairs_table(trades_with_all_nan_reasons)

        assert pairs.empty
        assert list(pairs.columns) == PAIRS_COLUMNS

    def test_partial_reasons_requires_both(self, trades_with_partial_reasons):
        pairs = calculate_entry_exit_pairs_table(trades_with_partial_reasons)

        assert len(pairs) == 1
        assert pairs.iloc[0]["Entry Reason -> Exit Reason"] == "manual -> take_profit"
