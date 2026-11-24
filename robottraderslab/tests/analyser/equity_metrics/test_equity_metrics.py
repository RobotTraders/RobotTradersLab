import pandas as pd
import pytest

from robottraderslab.analyser.equity_metrics.equity_metrics import (
    calculate_calmar_ratio,
    calculate_drawdowns,
    calculate_hodl_performance,
    calculate_performance_vs_hodl,
    calculate_returns,
    calculate_roi,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)


@pytest.fixture
def equity_series() -> pd.Series:
    """Create a sample equity curve for testing with multiple drawdowns."""
    dates = pd.date_range(start="2024-01-01", periods=13, freq="D")
    return pd.Series(
        [
            1000.0,
            1100.0,
            1050.0,
            1200.0,
            1150.0,
            1300.0,
            1250.0,
            1150.0,
            1100.0,
            1250.0,
            1400.0,
            1380.0,
            1450.0,
        ],
        index=dates,
        name="equity",
    )


@pytest.fixture
def returns_series() -> pd.Series:
    """Create a sample returns series for testing with various patterns."""
    return pd.Series(
        [
            0.1,
            -0.05,
            0.15,
            -0.04,
            0.13,
            -0.038,
            -0.08,
            -0.043,
            0.136,
            0.12,
            -0.014,
            0.051,
        ],
        name="returns",
    )


def test_calculate_returns_basic(equity_series):
    returns = calculate_returns(equity_series)
    expected_returns = pd.Series(
        [
            0.1,
            -0.0454545,
            0.1428571,
            -0.0416667,
            0.1304348,
            -0.0384615,
            -0.08,
            -0.0434783,
            0.1363636,
            0.12,
            -0.0142857,
            0.0507246,
        ],
        index=equity_series.index[1:],
        name="equity",
    )
    pd.testing.assert_series_equal(
        returns, expected_returns, rtol=1e-4, check_names=False
    )


@pytest.mark.parametrize(
    ("equity_values", "expected_roi"),
    [
        ([1000, 1100], 0.1),
        ([1000, 900], -0.1),
        ([1000, 1000], 0.0),
    ],
)
def test_calculate_roi(equity_values, expected_roi):
    equity = pd.Series(equity_values)
    assert calculate_roi(equity, equity_values[0]) == pytest.approx(expected_roi)


class TestCalculateHodlPerformance:
    @pytest.mark.parametrize(
        ("initial_capital", "start_price", "end_price", "expected_return"),
        [
            (1000, 100, 110, 0.1),
            (1000, 100, 90, -0.1),
            (1000, 100, 100, 0.0),
        ],
    )
    def test_with_various_price_scenarios(
        self, initial_capital, start_price, end_price, expected_return
    ):
        hodl = calculate_hodl_performance(initial_capital, start_price, end_price)

        assert hodl == pytest.approx(expected_return)


def test_calculate_drawdowns(equity_series):
    drawdowns = calculate_drawdowns(equity_series)

    assert drawdowns.max_drawdown == pytest.approx(-0.1538462, rel=1e-4)
    assert drawdowns.max_drawdown_amount == pytest.approx(-200.0)

    expected_dd_pct = pd.Series(
        [
            0.0,
            0.0,
            -0.0454545,
            0.0,
            -0.0416667,
            0.0,
            -0.0384615,
            -0.1153846,
            -0.1538462,
            -0.0384615,
            0.0,
            -0.0142857,
            0.0,
        ],
        index=equity_series.index,
        name="equity",
    )
    pd.testing.assert_series_equal(
        drawdowns.drawdown_percentage, expected_dd_pct, rtol=1e-4
    )

    expected_abs_dd = expected_dd_pct * equity_series.cummax()
    pd.testing.assert_series_equal(
        drawdowns.absolute_drawdown, expected_abs_dd, rtol=1e-4
    )


class TestCalculateSharpeRatio:
    def test_with_typical_market_scenario(self):
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        sharpe = calculate_sharpe_ratio(
            returns, annualization_factor=252, risk_free_rate=0.0
        )

        assert sharpe == pytest.approx(4.59, rel=1e-2)

    def test_with_strong_upward_trend(self):
        returns = pd.Series([0.03, 0.02, 0.04, 0.01, 0.03])
        sharpe = calculate_sharpe_ratio(
            returns, annualization_factor=252, risk_free_rate=0.0
        )

        assert sharpe == pytest.approx(36.19, rel=1e-2)

    def test_with_risk_free_rate(self):
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        risk_free = 0.02
        sharpe = calculate_sharpe_ratio(
            returns, annualization_factor=252, risk_free_rate=risk_free
        )
        assert sharpe == pytest.approx(4.533, rel=1e-2)

    def test_with_zero_annualization_factor(self):
        returns = pd.Series([0.05, -0.02, 0.03, -0.01, 0.04, 0.02])
        risk_free = 0.001667
        sharpe = calculate_sharpe_ratio(
            returns, annualization_factor=0, risk_free_rate=risk_free
        )
        assert sharpe == pytest.approx(0.598, rel=1e-2)


class TestCalculateSortinoRatio:
    def test_with_mixed_returns_upside_bias(self):
        returns = pd.Series([0.03, -0.01, 0.04, -0.02, 0.03])
        sortino = calculate_sortino_ratio(
            returns, annualization_factor=252, risk_free_rate=0.0
        )

        assert sortino == pytest.approx(22.224, rel=1e-3)

    def test_with_only_positive_returns(self):
        returns = pd.Series([0.02, 0.03, 0.01, 0.02])
        sortino = calculate_sortino_ratio(
            returns, annualization_factor=252, risk_free_rate=0.0
        )

        assert sortino == 0.0

    def test_with_risk_free_rate(self):
        returns = pd.Series([0.03, -0.01, 0.04, -0.02, 0.03])
        risk_free = 0.02
        sortino = calculate_sortino_ratio(
            returns, annualization_factor=252, risk_free_rate=risk_free
        )
        assert sortino == pytest.approx(21.996, rel=1e-2)

    def test_with_zero_annualization_factor(self):
        returns = pd.Series([0.05, -0.02, 0.03, -0.01, 0.04, 0.02])
        risk_free = 0.001667
        sortino = calculate_sortino_ratio(
            returns, annualization_factor=0, risk_free_rate=risk_free
        )
        assert sortino == pytest.approx(1.659, rel=1e-2)


class TestCalculateCalmarRatio:
    def test_with_typical_market_scenario(self):
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        max_drawdown = 0.05
        calmar = calculate_calmar_ratio(returns, max_drawdown, 252)

        assert calmar == pytest.approx(66.513, rel=1e-3)

    def test_with_strong_bull_market(self):
        returns = pd.Series([0.03, 0.02, 0.04, 0.01, 0.03])
        max_drawdown = 0.02
        calmar = calculate_calmar_ratio(returns, max_drawdown, 252)

        assert calmar == pytest.approx(31769.141, rel=1e-3)

    def test_with_bear_market(self):
        returns = pd.Series([-0.02, -0.01, -0.03, -0.01, -0.02])
        max_drawdown = 0.15
        calmar = calculate_calmar_ratio(returns, max_drawdown, 252)

        assert calmar == pytest.approx(-6.599, rel=1e-3)

    def test_with_zero_drawdown(self):
        returns = pd.Series([0.001] * 10)
        calmar = calculate_calmar_ratio(returns, 0.0, 252)

        assert calmar == 0.0

    def test_with_different_annualization_factor(self):
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        max_drawdown = 0.05
        calmar = calculate_calmar_ratio(returns, max_drawdown, 12)

        assert calmar == pytest.approx(1.44, rel=1e-2)

    def test_with_zero_annualization_factor(self):
        returns = pd.Series([0.05, -0.02, 0.03, -0.01, 0.04, 0.02])
        max_drawdown = 0.05
        calmar = calculate_calmar_ratio(returns, max_drawdown, annualization_factor=0)
        expected = returns.mean() / 0.05
        assert calmar == pytest.approx(expected, rel=1e-3)


class TestCalculatePerformanceVsHodl:
    def test_strategy_outperforms_hodl(self):
        equity_series = pd.Series([1000, 1200, 1300, 1500])
        price_series = pd.Series([100, 110, 120, 130])

        hodl_return, performance_vs_hodl = calculate_performance_vs_hodl(
            equity_series, price_series, 1000.0
        )

        assert hodl_return == pytest.approx(0.3, rel=1e-3)

        assert performance_vs_hodl == pytest.approx(0.1538, rel=1e-3)

    def test_strategy_underperforms_hodl(self):
        equity_series = pd.Series([1000, 1050, 1100, 1150])
        price_series = pd.Series([100, 110, 125, 140])

        hodl_return, performance_vs_hodl = calculate_performance_vs_hodl(
            equity_series, price_series, 1000.0
        )

        assert hodl_return == pytest.approx(0.4, rel=1e-3)

        assert performance_vs_hodl == pytest.approx(-0.1786, rel=1e-3)

    def test_strategy_matches_hodl_exactly(self):
        equity_series = pd.Series([1000, 1100, 1200, 1300])
        price_series = pd.Series([100, 110, 120, 130])

        hodl_return, performance_vs_hodl = calculate_performance_vs_hodl(
            equity_series, price_series, 1000.0
        )

        assert hodl_return == pytest.approx(0.3, rel=1e-3)

        assert performance_vs_hodl == pytest.approx(0.0, rel=1e-3)

    def test_negative_returns_for_both(self):
        equity_series = pd.Series([1000, 900, 850, 800])
        price_series = pd.Series([100, 90, 85, 80])

        hodl_return, performance_vs_hodl = calculate_performance_vs_hodl(
            equity_series, price_series, 1000.0
        )

        assert hodl_return == pytest.approx(-0.2, rel=1e-3)

        assert performance_vs_hodl == pytest.approx(0.0, rel=1e-3)

    def test_with_empty_series(self):
        empty_equity = pd.Series([], dtype=float)
        empty_price = pd.Series([], dtype=float)

        hodl_return, performance_vs_hodl = calculate_performance_vs_hodl(
            empty_equity, empty_price, 0.0
        )

        assert hodl_return == 0.0
        assert performance_vs_hodl == 0.0

    def test_price_crash_scenario_strategy_protects_capital(self):
        equity_series = pd.Series([1000, 1000, 950, 900])
        price_series = pd.Series([100, 90, 60, 40])

        hodl_return, performance_vs_hodl = calculate_performance_vs_hodl(
            equity_series, price_series, 1000.0
        )

        assert hodl_return == pytest.approx(-0.6, rel=1e-3)

        assert performance_vs_hodl == pytest.approx(1.25, rel=1e-3)
