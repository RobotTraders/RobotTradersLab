from unittest.mock import Mock

import pandas as pd
import pytest

from robottraderslab.analyser import Analyser
from robottraderslab.analyser.equity_metrics import EquityMetricsResult
from robottraderslab.analyser.trade_metrics import (
    TradeDrawdownMetrics,
    TradeMetricsResult,
)
from robottraderslab.chart import ChartService
from robottraderslab.plotting import PlottingService


@pytest.fixture
def mock_metrics() -> tuple[Mock, Mock]:
    trade_metrics = Mock(spec=TradeMetricsResult)
    trade_metrics.win_rate = 0.667
    trade_metrics.risk_reward_ratio = 2.15
    trade_metrics.total_trades = 10
    equity_metrics = Mock(spec=EquityMetricsResult)
    equity_metrics.sharpe_ratio = 1.85
    equity_metrics.roi = 0.092
    equity_metrics.max_drawdown = -0.037
    equity_metrics.max_drawdown_amount = -37.0
    equity_metrics.period_start = pd.Timestamp("2024-01-01")
    equity_metrics.period_end = pd.Timestamp("2024-01-02")
    equity_metrics.initial_equity = 1000.0
    equity_metrics.final_equity = 1010.0
    equity_metrics.net_profit = 10.0
    return trade_metrics, equity_metrics


def _build_analyser(
    trade_metrics: Mock,
    equity_metrics: Mock,
    equity_curve: pd.Series | None = None,
    **kwargs: object,
) -> Analyser:
    if equity_curve is None:
        equity_curve = pd.Series(dtype=float)
    return Analyser(
        trades=pd.DataFrame(),
        open_positions=pd.DataFrame(),
        equity_curve=equity_curve,
        trade_metrics=trade_metrics,
        equity_metrics=equity_metrics,
        trade_drawdown=TradeDrawdownMetrics(max_drawdown=-0.02, amount=-20.2),
        plotting_service=Mock(spec=PlottingService),
        chart_service=Mock(spec=ChartService),
        **kwargs,
    )


class TestSummaryMetrics:
    def test_returns_metrics_from_precomputed_results(self, mock_metrics):
        trade_metrics, equity_metrics = mock_metrics
        analyser = _build_analyser(trade_metrics, equity_metrics)

        summary = analyser.get_summary_metrics()

        assert summary.sharpe_ratio == 1.85
        assert summary.roi == 0.092
        assert summary.max_drawdown == -0.037
        assert summary.win_rate == 0.667
        assert summary.risk_reward_ratio == 2.15


class TestAnalyserWithoutSavePath:
    def test_prints_to_stdout(self, mock_metrics, capsys):
        trade_metrics, equity_metrics = mock_metrics
        equity_metrics.hodls = []
        trade_metrics.time_in_position_ratio = 0.5
        trade_metrics.profit_factor = 2.0
        trade_metrics.max_drawdown_at_trades = -0.02
        trade_metrics.max_drawdown_at_trades_amount = -20.2
        trade_metrics.avg_trade_pnl = 0.50
        trade_metrics.avg_trade_return = 0.005
        trade_metrics.avg_trades_per_day = 1.0
        trade_metrics.avg_trade_duration_days = 0.5
        trade_metrics.avg_winning_trade_pnl = 1.00
        trade_metrics.avg_winning_trade_return = 0.01
        trade_metrics.avg_losing_trade_pnl = -0.50
        trade_metrics.avg_losing_trade_return = -0.005
        trade_metrics.avg_winning_trade_duration_days = 0.3
        trade_metrics.avg_losing_trade_duration_days = 0.7
        trade_metrics.largest_winning_trade_pnl = 3.00
        trade_metrics.largest_losing_trade_pnl = 1.50
        trade_metrics.best_trade_return = 0.02
        trade_metrics.worst_trade_return = -0.015
        trade_metrics.max_win_streak = 3
        trade_metrics.max_lose_streak = 2
        trade_metrics.winning_trades = 6
        trade_metrics.losing_trades = 4
        trade_metrics.open_reasons = pd.Series(dtype=str)
        trade_metrics.close_reasons = pd.Series(dtype=str)
        trade_metrics.total_fee = 10.0
        trade_metrics.biggest_fee = 2.0
        trade_metrics.avg_fee = 1.0
        trade_metrics.total_pnl = 5.0
        equity_metrics.return_over_max_drawdown = 2.5
        equity_metrics.sharpe_ratio = 1.85
        equity_metrics.sortino_ratio = 2.1
        equity_metrics.calmar_ratio = 1.5

        equity_curve = pd.Series(
            [1000.0, 1010.0],
            index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
        )
        analyser = _build_analyser(trade_metrics, equity_metrics, equity_curve)

        analyser.print_performance_summary()

        captured = capsys.readouterr()
        assert "PnL:" in captured.out
