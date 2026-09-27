import pandas as pd
import pytest

from robottraderslab.analyser import Analyser
from robottraderslab.backtester import BacktestOutputs


class TestBacktestPipelineBasics:
    """Verify the backtester pipeline runs end-to-end and produces correct outputs."""

    def test_backtest_runs_successfully(self, backtest_outputs):
        assert backtest_outputs.final_balance["USDT"] > 0
        assert len(backtest_outputs.get_equity_curve()) > 0
        assert len(backtest_outputs.fills) > 0

    def test_generates_long_entry_and_exit_transactions(self, backtest_outputs):
        transactions = backtest_outputs.fills

        entry_fills = transactions[transactions["fill_type"] == "enter_long"]
        exit_fills = transactions[transactions["fill_type"] == "exit_long"]

        assert len(entry_fills) > 0
        assert len(exit_fills) > 0
        assert abs(len(entry_fills) - len(exit_fills)) <= 1

    def test_equity_curve_ends_at_final_balance(self, backtest_outputs):
        equity_curve = backtest_outputs.get_equity_curve()

        assert len(equity_curve) > 100
        assert (
            abs(equity_curve.iloc[-1] - backtest_outputs.final_balance["USDT"]) < 0.01
        )


class TestBacktestPipelineEquityMetrics:
    """Verify the backtester pipeline computes correct equity metrics."""

    EXPECTED_FINAL_BALANCE = 15580.935525008175
    EXPECTED_ROI = 0.5580935525
    EXPECTED_MAX_DRAWDOWN = -0.3649327571
    EXPECTED_SHARPE_RATIO = 1.27
    EXPECTED_SORTINO_RATIO = 2.02
    EXPECTED_CALMAR_RATIO = 1.53
    EXPECTED_ROMAD_RATIO = 1.5293052102
    EXPECTED_DRAWDOWN_SERIES_LENGTH = 367
    EXPECTED_ABSOLUTE_DRAWDOWN_SERIES_LENGTH = 367
    EXPECTED_RETURNS_LENGTH = 366

    TOLERANCE = 0.01

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        backtest_outputs: BacktestOutputs,
        analyser: Analyser,
    ) -> None:
        self.final_balance = backtest_outputs.final_balance["USDT"]
        self.equity_curve = backtest_outputs.get_equity_curve()
        self.analyser = analyser

    def test_final_balance(self):
        assert abs(self.final_balance - self.EXPECTED_FINAL_BALANCE) < self.TOLERANCE

    def test_roi(self):
        assert abs(self.analyser.roi - self.EXPECTED_ROI) < self.TOLERANCE

    def test_max_drawdown(self):
        assert (
            abs(self.analyser.max_drawdown - self.EXPECTED_MAX_DRAWDOWN)
            < self.TOLERANCE
        )

    def test_sharpe_ratio(self):
        assert (
            abs(self.analyser.sharpe_ratio - self.EXPECTED_SHARPE_RATIO)
            < self.TOLERANCE
        )

    def test_sortino_ratio(self):
        assert (
            abs(self.analyser.sortino_ratio - self.EXPECTED_SORTINO_RATIO)
            < self.TOLERANCE
        )

    def test_calmar_ratio(self):
        assert (
            abs(self.analyser.calmar_ratio - self.EXPECTED_CALMAR_RATIO)
            < self.TOLERANCE
        )

    def test_romad_ratio(self):
        assert (
            abs(self.analyser.romad_ratio - self.EXPECTED_ROMAD_RATIO) < self.TOLERANCE
        )

    def test_drawdown_series(self):
        drawdown_series = self.analyser.drawdown_series

        assert len(drawdown_series) == self.EXPECTED_DRAWDOWN_SERIES_LENGTH
        assert isinstance(drawdown_series, pd.Series)
        assert (drawdown_series <= 0).all()
        assert abs(drawdown_series.min() - self.EXPECTED_MAX_DRAWDOWN) < self.TOLERANCE

    def test_absolute_drawdown_series(self):
        absolute_drawdown_series = self.analyser.absolute_drawdown_series

        assert (
            len(absolute_drawdown_series)
            == self.EXPECTED_ABSOLUTE_DRAWDOWN_SERIES_LENGTH
        )
        assert isinstance(absolute_drawdown_series, pd.Series)
        assert (absolute_drawdown_series <= 0).all()

    def test_returns_series(self):
        returns = self.analyser.returns

        assert len(returns) == self.EXPECTED_RETURNS_LENGTH
        assert isinstance(returns, pd.Series)


class TestBacktestPipelineTradeMetrics:
    """Verify the backtester pipeline computes correct trade metrics."""

    EXPECTED_CLOSED_TRADES_COUNT = 6
    EXPECTED_WINNING_TRADES_COUNT = 2
    EXPECTED_LOSING_TRADES_COUNT = 4
    EXPECTED_WIN_RATE = 0.3333333333
    EXPECTED_PROFIT_FACTOR = 2.40392488724539
    EXPECTED_MAX_WIN_STREAK = 1
    EXPECTED_MAX_LOSE_STREAK = 4
    EXPECTED_TOTAL_FEES = 154.22981623342508
    EXPECTED_AVERAGE_FEE = 25.704969372237514
    EXPECTED_BIGGEST_FEE = 29.646215829076
    EXPECTED_AVG_TRADE_PNL = 930.1559208346963
    EXPECTED_AVG_WINNING_TRADE_PNL = 4778.086749713017
    EXPECTED_AVG_LOSING_TRADE_PNL = 993.8094936044637
    EXPECTED_LARGEST_WINNING_TRADE_PNL = 4907.001920068786
    EXPECTED_LARGEST_LOSING_TRADE_PNL = 2374.5540314384025
    EXPECTED_AVG_TRADE_DURATION_DAYS = 36.3333333333
    EXPECTED_AVG_WINNING_TRADE_DURATION_DAYS = 71.5000000000
    EXPECTED_AVG_LOSING_TRADE_DURATION_DAYS = 18.7500000000
    EXPECTED_TOTAL_PNL = 5580.935525008178

    TOLERANCE = 1e-8

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        analyser: Analyser,
    ) -> None:
        self.analyser = analyser

    def test_closed_trades_count(self):
        assert self.analyser.closed_trades_count == self.EXPECTED_CLOSED_TRADES_COUNT

    def test_winning_trades_count(self):
        assert self.analyser.winning_trades_count == self.EXPECTED_WINNING_TRADES_COUNT

    def test_losing_trades_count(self):
        assert self.analyser.losing_trades_count == self.EXPECTED_LOSING_TRADES_COUNT

    def test_win_rate(self):
        assert abs(self.analyser.win_rate - self.EXPECTED_WIN_RATE) < self.TOLERANCE

    def test_profit_factor(self):
        assert (
            abs(self.analyser.profit_factor - self.EXPECTED_PROFIT_FACTOR)
            < self.TOLERANCE
        )

    def test_max_win_streak(self):
        assert self.analyser.max_win_streak == self.EXPECTED_MAX_WIN_STREAK

    def test_max_lose_streak(self):
        assert self.analyser.max_lose_streak == self.EXPECTED_MAX_LOSE_STREAK

    def test_total_fees(self):
        assert abs(self.analyser.total_fees - self.EXPECTED_TOTAL_FEES) < self.TOLERANCE

    def test_average_fee(self):
        assert (
            abs(self.analyser.average_fee - self.EXPECTED_AVERAGE_FEE) < self.TOLERANCE
        )

    def test_biggest_fee(self):
        assert (
            abs(self.analyser.biggest_fee - self.EXPECTED_BIGGEST_FEE) < self.TOLERANCE
        )

    def test_avg_trade_pnl(self):
        assert (
            abs(self.analyser.avg_trade_pnl - self.EXPECTED_AVG_TRADE_PNL)
            < self.TOLERANCE
        )

    def test_avg_winning_trade_pnl(self):
        assert (
            abs(
                self.analyser.avg_winning_trade_pnl
                - self.EXPECTED_AVG_WINNING_TRADE_PNL
            )
            < self.TOLERANCE
        )

    def test_avg_losing_trade_pnl(self):
        assert (
            abs(self.analyser.avg_losing_trade_pnl - self.EXPECTED_AVG_LOSING_TRADE_PNL)
            < self.TOLERANCE
        )

    def test_largest_winning_trade_pnl(self):
        assert (
            abs(
                self.analyser.largest_winning_trade_pnl
                - self.EXPECTED_LARGEST_WINNING_TRADE_PNL
            )
            < self.TOLERANCE
        )

    def test_largest_losing_trade_pnl(self):
        assert (
            abs(
                self.analyser.largest_losing_trade_pnl
                - self.EXPECTED_LARGEST_LOSING_TRADE_PNL
            )
            < self.TOLERANCE
        )

    def test_avg_trade_duration_days(self):
        assert (
            abs(
                self.analyser.avg_trade_duration_days
                - self.EXPECTED_AVG_TRADE_DURATION_DAYS
            )
            < self.TOLERANCE
        )

    def test_avg_winning_trade_duration_days(self):
        assert (
            abs(
                self.analyser.avg_winning_trade_duration_days
                - self.EXPECTED_AVG_WINNING_TRADE_DURATION_DAYS
            )
            < self.TOLERANCE
        )

    def test_avg_losing_trade_duration_days(self):
        assert (
            abs(
                self.analyser.avg_losing_trade_duration_days
                - self.EXPECTED_AVG_LOSING_TRADE_DURATION_DAYS
            )
            < self.TOLERANCE
        )

    def test_total_pnl(self):
        assert abs(self.analyser.total_pnl - self.EXPECTED_TOTAL_PNL) < self.TOLERANCE
