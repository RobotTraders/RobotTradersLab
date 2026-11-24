import pandas as pd
import pytest

from robottraderslab.analyser import TradeFilter
from robottraderslab.analyser.trade_metrics import compute_trade_metrics


@pytest.fixture
def two_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "net_pnl": [100.0, -50.0],
            "net_pnl_pct": [0.1, -0.05],
            "entry_fee": [1.0, 2.0],
            "exit_fee": [1.0, 2.0],
            "entry_time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "exit_time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "entry_reason": ["signal", "signal"],
            "exit_reason": ["tp", "sl"],
            "side": ["long", "short"],
            "symbol": ["BTC/USDT:USDT", "BTC/USDT:USDT"],
        }
    )


class TestComputeTradeMetrics:
    def test_a_run_without_trades_measures_to_zero(self, two_trades):
        metrics = compute_trade_metrics(two_trades.iloc[0:0])

        assert metrics.total_trades == 0
        assert metrics.total_pnl == 0.0
        assert metrics.win_rate == 0.0

    def test_the_whole_run_carries_the_empty_filter(self, two_trades):
        metrics = compute_trade_metrics(two_trades)

        assert metrics.filter_applied == TradeFilter()
        assert metrics.total_trades == 2

    def test_a_filter_keeps_only_its_trades(self, two_trades):
        metrics = compute_trade_metrics(two_trades, TradeFilter(side="long"))

        assert metrics.total_trades == 1
        assert metrics.total_pnl == 100.0
        assert metrics.filter_applied == TradeFilter(side="long")

    def test_a_filter_keeping_nothing_names_itself(self, two_trades):
        with pytest.raises(ValueError, match="symbol_ETH_USDT_USDT"):
            compute_trade_metrics(two_trades, TradeFilter(symbol="ETH/USDT:USDT"))
