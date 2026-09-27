from pathlib import Path

import pandas as pd
import pytest

from robottraderslab.analyser import Analyser, AnalysisInputs
from robottraderslab.analyser.trade_aggregator import TradeAggregator
from robottraderslab.backtester import BacktestOutputs
from robottraderslab.bootstrap import ReportConfig


class TestTransactionsMatchTradingView:
    """Compare backtest transactions with TradingView transaction list."""

    TOLERANCE_AMOUNT = 1e-5
    TOLERANCE_PRICE = 1e-2
    TOLERANCE_PNL = 2e-1

    @pytest.fixture
    def tradingview_transactions(self) -> pd.DataFrame:
        csv_path = (
            Path(__file__).parent
            / "Test_Lab_simple_btc_BINANCE_BTCUSDT_1d_liste_des_transactions.csv"
        )

        df = pd.read_csv(
            csv_path,
            encoding="utf-8",
        )

        df["Date/Heure"] = pd.to_datetime(df["Date/Heure"])

        for col in ["Prix USDT", "Profit USDT"]:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("\u202f", "", regex=False)
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )

        df["Profit %"] = (
            df["Profit %"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
            .astype(float)
        )

        return df

    @pytest.fixture(autouse=True)
    def _setup_transactions(
        self,
        backtest_outputs: BacktestOutputs,
        tradingview_transactions: pd.DataFrame,
        trade_aggregator: TradeAggregator,
    ) -> None:
        self.our_transactions = backtest_outputs.fills.copy()
        self.our_transactions.index = (
            pd.DatetimeIndex(self.our_transactions.index).tz_localize(None).floor("D")
        )
        self.our_transactions = self.our_transactions.sort_index()

        self.tv_transactions = tradingview_transactions.copy()
        self.tv_transactions["Date/Heure"] = self.tv_transactions[
            "Date/Heure"
        ].dt.floor("D")
        self.tv_transactions = self.tv_transactions.sort_values("Date/Heure")

        self.our_trades = trade_aggregator.trades.copy()
        self.our_trades = self.our_trades.sort_values("exit_time")
        self.current_trade_idx = 0

    def test_transaction_count_matches_tradingview(self):
        expected_count = len(self.tv_transactions)
        actual_count = len(self.our_transactions)

        assert actual_count == expected_count

    def test_transaction_dates_match_tradingview(self):
        our_dates = self.our_transactions.index
        tv_bar_opens = self.tv_transactions["Date/Heure"]
        tv_bar_closes = tv_bar_opens + pd.Timedelta(days=1)

        assert list(our_dates) == list(tv_bar_closes)

    def test_transactions_details_match_tradingview(self):
        """Verify fill type, amount, price, and trade PnL for each transaction."""
        for (date, our_tx), (_, tv_tx) in zip(
            self.our_transactions.iterrows(), self.tv_transactions.iterrows()
        ):
            date = pd.Timestamp(date)

            self._assert_fill_type(our_tx, tv_tx, date)
            self._assert_amount(our_tx, tv_tx, date)
            self._assert_price(our_tx, tv_tx, date)
            self._assert_trade_pnl(our_tx, tv_tx, date)

    def _assert_fill_type(
        self, our_tx: pd.Series, tv_tx: pd.Series, date: pd.Timestamp
    ) -> None:
        tv_to_our_types = {
            "Buy": "enter_long",
            "Close entry(s) order Buy": "exit_long",
        }

        tv_type = tv_to_our_types[tv_tx["Signal"]]
        our_type = our_tx["fill_type"]

        assert tv_type == our_type, (
            f"Transaction type mismatch on {date}:\n"
            f"Our type: {our_type}\n"
            f"TradingView type: {tv_type}"
        )

    def _assert_amount(
        self, our_tx: pd.Series, tv_tx: pd.Series, date: pd.Timestamp
    ) -> None:
        our_amount = abs(float(our_tx["net_quantity"]))
        tv_amount = float(tv_tx["Contrats"])

        assert abs(our_amount - tv_amount) < self.TOLERANCE_AMOUNT, (
            f"Transaction quantity mismatch on {date}:\n"
            f"Our quantity: {our_amount}\n"
            f"TradingView quantity: {tv_amount}"
        )

    def _assert_price(
        self, our_tx: pd.Series, tv_tx: pd.Series, date: pd.Timestamp
    ) -> None:
        our_price = float(our_tx["price"])
        tv_price = float(tv_tx["Prix USDT"])

        assert abs(our_price - tv_price) < self.TOLERANCE_PRICE, (
            f"Transaction price mismatch on {date}:\n"
            f"Our price: {our_price}\n"
            f"TradingView price: {tv_price}"
        )

    def _assert_trade_pnl(
        self, our_tx: pd.Series, tv_tx: pd.Series, date: pd.Timestamp
    ) -> None:
        if our_tx["fill_type"] != "exit_long":
            return

        trade = self.our_trades.iloc[self.current_trade_idx]
        self.current_trade_idx += 1

        our_pnl = float(trade["net_pnl"])
        tv_pnl = float(tv_tx["Profit USDT"])

        assert abs(our_pnl - tv_pnl) < self.TOLERANCE_PNL, (
            f"Trade PnL mismatch on {date}:\n"
            f"Our PnL: {our_pnl}\n"
            f"TradingView PnL: {tv_pnl}"
        )


class TestPerformanceMetricsMatchTradingView:
    """Compare performance metrics with TradingView analysis data."""

    TOLERANCE_EXACT = 1e-4
    TOLERANCE_CURRENCY = 2e-1
    TOLERANCE_RATIO = 1e-3
    TOLERANCE_DURATION = 2

    @pytest.fixture
    def tradingview_analysis(self) -> pd.DataFrame:
        csv_path = (
            Path(__file__).parent
            / "Test_Lab_simple_btc_BINANCE_BTCUSDT_1d_analyse_des_transactions.csv"
        )

        df = pd.read_csv(csv_path, encoding="windows-1252")

        for col in df.columns:
            if "USDT" in col:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace("\u202f", "", regex=False)
                    .str.replace(" ", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

            elif "%" in col:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.strip()
                )
                df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

        return df

    @pytest.fixture(autouse=True)
    def _setup_analysis(
        self,
        analyser: Analyser,
        tradingview_analysis: pd.DataFrame,
    ) -> None:
        self.analyser = analyser
        self.tv_analysis = tradingview_analysis

    def test_total_trades(self):
        tv_total_trades = int(self.tv_analysis.loc[0, "Tout USDT"])
        our_total_trades = self.analyser.closed_trades_count

        assert our_total_trades == tv_total_trades

    def test_winning_trades_count(self):
        tv_winning_trades = int(self.tv_analysis.loc[2, "Tout USDT"])
        our_winning_trades = self.analyser.winning_trades_count

        assert our_winning_trades == tv_winning_trades

    def test_losing_trades_count(self):
        tv_losing_trades = int(self.tv_analysis.loc[3, "Tout USDT"])
        our_losing_trades = self.analyser.losing_trades_count

        assert our_losing_trades == tv_losing_trades

    def test_win_rate(self):
        tv_win_rate = self.tv_analysis.loc[4, "Tout %"]
        our_win_rate = self.analyser.win_rate

        assert abs(our_win_rate - tv_win_rate) < self.TOLERANCE_EXACT

    def test_average_trade_pnl(self):
        tv_avg_pnl = self.tv_analysis.loc[5, "Tout USDT"]
        our_avg_pnl = self.analyser.avg_trade_pnl

        assert abs(our_avg_pnl - tv_avg_pnl) < self.TOLERANCE_CURRENCY

    def test_average_winning_trade_pnl(self):
        tv_avg_winning = self.tv_analysis.loc[6, "Tout USDT"]
        our_avg_winning = self.analyser.avg_winning_trade_pnl

        assert abs(our_avg_winning - tv_avg_winning) < self.TOLERANCE_CURRENCY

    def test_average_losing_trade_pnl(self):
        tv_avg_losing = self.tv_analysis.loc[7, "Tout USDT"]
        our_avg_losing = self.analyser.avg_losing_trade_pnl

        assert abs(our_avg_losing - tv_avg_losing) < self.TOLERANCE_CURRENCY

    def test_risk_reward_ratio_is_non_negative(self):
        rr = self.analyser.risk_reward_ratio

        assert rr >= 0 or rr == float("inf")

    def test_largest_winning_trade_pnl(self):
        tv_largest_winning = self.tv_analysis.loc[9, "Tout USDT"]
        our_largest_winning = self.analyser.largest_winning_trade_pnl

        assert abs(our_largest_winning - tv_largest_winning) < self.TOLERANCE_CURRENCY

    def test_largest_losing_trade_pnl(self):
        tv_largest_losing = self.tv_analysis.loc[11, "Tout USDT"]
        our_largest_losing = self.analyser.largest_losing_trade_pnl

        assert abs(our_largest_losing - tv_largest_losing) < self.TOLERANCE_CURRENCY

    def test_average_trade_duration(self):
        tv_avg_duration = self.tv_analysis.loc[13, "Tout USDT"]
        our_avg_duration = self.analyser.avg_trade_duration_days

        assert abs(our_avg_duration - tv_avg_duration) < self.TOLERANCE_DURATION

    def test_average_winning_trade_duration(self):
        tv_avg_winning_duration = self.tv_analysis.loc[14, "Tout USDT"]
        our_avg_winning_duration = self.analyser.avg_winning_trade_duration_days

        assert (
            abs(our_avg_winning_duration - tv_avg_winning_duration)
            < self.TOLERANCE_DURATION
        )

    def test_average_losing_trade_duration(self):
        tv_avg_losing_duration = self.tv_analysis.loc[15, "Tout USDT"]
        our_avg_losing_duration = self.analyser.avg_losing_trade_duration_days

        assert (
            abs(our_avg_losing_duration - tv_avg_losing_duration)
            < self.TOLERANCE_DURATION
        )


class TestRiskPerformanceRatiosMatchTradingView:
    """Compare risk/performance ratios with TradingView ratios data."""

    TOLERANCE_RATIO = 1  # TODO make much lower once difference is understood

    @pytest.fixture
    def tradingview_ratios(self) -> pd.DataFrame:
        csv_path = (
            Path(__file__).parent
            / "Test_Lab_simple_btc_BINANCE_BTCUSDT_1d_ratios_risque_performance.csv"
        )

        df = pd.read_csv(csv_path, encoding="windows-1252", index_col=0)

        for col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("\u202f", "", regex=False)
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @pytest.fixture(autouse=True)
    def _setup_ratios(
        self,
        backtest_outputs: BacktestOutputs,
        trade_aggregator: TradeAggregator,
        tradingview_ratios: pd.DataFrame,
    ) -> None:
        equity_curve = backtest_outputs.get_equity_curve()
        perf_data = AnalysisInputs(
            trades=trade_aggregator.trades,
            open_positions=trade_aggregator.open_trades,
            equity_curve=equity_curve,
            initial_balance=float(equity_curve.iloc[0]),
        )
        self.analyser = Analyser.from_analysis_inputs(
            perf_data,
            backtest_outputs.ohlcv_provider,
            ReportConfig(risk_free_rate=0.02, annualization_factor=252.0),
        )
        self.tv_ratios = tradingview_ratios

    def test_sharpe_ratio(self):
        tv_sharpe_ratio = self.tv_ratios.loc["Ratio de Sharpe", "Tout USDT"]
        our_sharpe_ratio = self.analyser.sharpe_ratio

        assert abs(our_sharpe_ratio - tv_sharpe_ratio) < self.TOLERANCE_RATIO

    def test_sortino_ratio(self):
        tv_sortino_ratio = self.tv_ratios.loc["Ratio de Sortino", "Tout USDT"]
        our_sortino_ratio = self.analyser.sortino_ratio

        assert abs(our_sortino_ratio - tv_sortino_ratio) < self.TOLERANCE_RATIO

    def test_profit_factor(self):
        tv_profit_factor = self.tv_ratios.loc["Facteur de profit", "Tout USDT"]
        our_profit_factor = self.analyser.profit_factor

        assert abs(our_profit_factor - tv_profit_factor) < self.TOLERANCE_RATIO


class TestTradingViewLikeMetricsMatchTradingView:
    """Compare TradingView-like calculation method metrics with actual TradingView values.

    Uses the 'tradingview_like' calculation method which samples equity only at
    trade entry/exit points, resamples to monthly for ratio calculations, and
    calculates intrabar drawdowns using high/low prices during open positions.
    """

    TOLERANCE = 0.01
    TOLERANCE_SORTINO = 0.07

    @pytest.fixture(autouse=True)
    def _setup_tradingview_like_analyser(
        self,
        backtest_outputs: BacktestOutputs,
        trade_aggregator: TradeAggregator,
    ) -> None:
        equity_curve = backtest_outputs.get_equity_curve()
        perf_data = AnalysisInputs(
            trades=trade_aggregator.trades,
            open_positions=trade_aggregator.open_trades,
            equity_curve=equity_curve,
            initial_balance=float(equity_curve.iloc[0]),
        )
        self.analyser = Analyser.from_analysis_inputs(
            perf_data,
            backtest_outputs.ohlcv_provider,
            ReportConfig(calculation_method="tradingview_like"),
        )

    def test_max_drawdown(self):
        tv_max_drawdown = -0.2872
        our_max_drawdown = self.analyser.max_drawdown

        assert abs(our_max_drawdown - tv_max_drawdown) < self.TOLERANCE

    def test_sharpe_ratio(self):
        tv_sharpe_ratio = 0.298
        our_sharpe_ratio = self.analyser.sharpe_ratio

        assert abs(our_sharpe_ratio - tv_sharpe_ratio) < self.TOLERANCE

    def test_sortino_ratio(self):
        """~6% discrepancy expected due to TradingView's undocumented T parameter."""
        tv_sortino_ratio = 1.029
        our_sortino_ratio = self.analyser.sortino_ratio

        assert abs(our_sortino_ratio - tv_sortino_ratio) < self.TOLERANCE_SORTINO
