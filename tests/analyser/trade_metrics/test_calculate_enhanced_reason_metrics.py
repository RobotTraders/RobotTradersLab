import pandas as pd
import pytest

from robottraderslab.analyser.trade_metrics import calculate_enhanced_reason_metrics


@pytest.fixture
def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_reason": ["signal", "signal", "manual", "signal", "breakout"],
            "exit_reason": [
                "take_profit",
                "stop_loss",
                "take_profit",
                "signal",
                "take_profit",
            ],
            "net_pnl": [100, -50, 75, -25, 150],
            "net_pnl_pct": [0.01, -0.005, 0.0075, -0.0025, 0.015],
        }
    )


class TestCalculateEnhancedReasonMetrics:
    def test_the_entry_reasons_are_counted(self, trades):
        entry_reasons = calculate_enhanced_reason_metrics(trades).entry_reasons

        assert entry_reasons.to_dict() == {"signal": 3, "manual": 1, "breakout": 1}

    def test_the_exit_reasons_are_counted(self, trades):
        exit_reasons = calculate_enhanced_reason_metrics(trades).exit_reasons

        assert exit_reasons.to_dict() == {"take_profit": 3, "stop_loss": 1, "signal": 1}

    def test_the_counts_are_named_after_their_column(self, trades):
        reasons = calculate_enhanced_reason_metrics(trades)

        assert reasons.entry_reasons.name == "entry_reason"
        assert reasons.exit_reasons.name == "exit_reason"

    def test_every_pair_of_reasons_gets_a_row(self, trades):
        pairs = calculate_enhanced_reason_metrics(trades).entry_exit_pairs_table

        assert len(pairs) == 5
