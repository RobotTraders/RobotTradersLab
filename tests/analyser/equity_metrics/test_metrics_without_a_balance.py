from unittest.mock import Mock

import pandas as pd
import pytest

from robottraderslab._core import OHLCVProviderProtocol
from robottraderslab.analyser.equity_metrics import compute_equity_metrics

_PROFIT_CURVE = [0.0, 40.0, 25.0, 60.0, 45.0]


@pytest.fixture
def profit_curve() -> pd.Series:
    """A curve tracking what trading earned, starting from nothing."""
    return pd.Series(
        _PROFIT_CURVE, index=pd.date_range("2024-01-01", periods=5, freq="D")
    )


@pytest.fixture
def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2024-01-01"]),
            "exit_time": pd.to_datetime(["2024-01-02"]),
            "entry_price": [100.0],
            "exit_price": [140.0],
            "net_quantity": [1.0],
            "side": ["long"],
            "symbol": ["BTC/USDT"],
            "net_pnl": [40.0],
        }
    )


@pytest.fixture
def ohlcv_provider() -> OHLCVProviderProtocol:
    return Mock(spec=OHLCVProviderProtocol)


def _measure(curve, trades, provider, balance=None):
    return compute_equity_metrics(curve, trades, provider, initial_balance=balance)


class TestFiguresStatedWithoutABalance:
    def test_the_profit_is_the_distance_the_curve_travelled(
        self, profit_curve, trades, ohlcv_provider
    ):
        metrics = _measure(profit_curve, trades, ohlcv_provider)

        assert metrics.net_profit == pytest.approx(45.0)

    def test_the_deepest_fall_is_stated_in_currency(
        self, profit_curve, trades, ohlcv_provider
    ):
        metrics = _measure(profit_curve, trades, ohlcv_provider)

        assert metrics.max_drawdown_amount == pytest.approx(-15.0)

    def test_the_fall_is_measured_against_the_curve_own_peak(
        self, profit_curve, trades, ohlcv_provider
    ):
        metrics = _measure(profit_curve, trades, ohlcv_provider)

        assert metrics.absolute_drawdown.min() == pytest.approx(-15.0)

    def test_the_period_still_spans_the_curve(
        self, profit_curve, trades, ohlcv_provider
    ):
        metrics = _measure(profit_curve, trades, ohlcv_provider)

        assert metrics.period_start == pd.Timestamp("2024-01-01")
        assert metrics.period_end == pd.Timestamp("2024-01-05")


class TestFiguresLeftUnstatedWithoutABalance:
    @pytest.mark.parametrize(
        "figure",
        [
            "initial_equity",
            "final_equity",
            "roi",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "return_over_max_drawdown",
            "drawdown_percentage",
            "returns",
        ],
    )
    def test_a_figure_measured_against_a_balance_is_absent(
        self, profit_curve, trades, ohlcv_provider, figure
    ):
        metrics = _measure(profit_curve, trades, ohlcv_provider)

        assert getattr(metrics, figure) is None

    def test_no_benchmark_is_compared_against(
        self, profit_curve, trades, ohlcv_provider
    ):
        metrics = _measure(profit_curve, trades, ohlcv_provider)

        assert metrics.hodls == []

    def test_the_tradingview_method_refuses_to_measure(
        self, profit_curve, trades, ohlcv_provider
    ):
        with pytest.raises(ValueError, match="initial balance"):
            compute_equity_metrics(
                profit_curve,
                trades,
                ohlcv_provider,
                calculation_method="tradingview_like",
            )


class TestABalanceRestoresTheRatios:
    def test_a_balance_yields_a_return(self, profit_curve, trades, ohlcv_provider):
        equity_curve = profit_curve + 1000.0

        metrics = _measure(equity_curve, trades, ohlcv_provider, balance=1000.0)

        assert metrics.roi == pytest.approx(0.045)

    def test_a_balance_yields_the_fall_as_a_share_of_the_peak(
        self, profit_curve, trades, ohlcv_provider
    ):
        equity_curve = profit_curve + 1000.0

        metrics = _measure(equity_curve, trades, ohlcv_provider, balance=1000.0)

        assert metrics.max_drawdown == pytest.approx(-15.0 / 1040.0)

    def test_the_currency_profit_is_the_same_either_way(
        self, profit_curve, trades, ohlcv_provider
    ):
        unmeasured = _measure(profit_curve, trades, ohlcv_provider)
        measured = _measure(
            profit_curve + 1000.0, trades, ohlcv_provider, balance=1000.0
        )

        assert measured.net_profit == pytest.approx(unmeasured.net_profit)


class TestAWindowTooShortToReachAMidnight:
    @pytest.fixture
    def intraday_curve(self) -> pd.Series:
        return pd.Series(
            [0.0, 100.0, 10.0, 30.0],
            index=pd.to_datetime(
                [
                    "2024-01-01 09:00",
                    "2024-01-01 10:00",
                    "2024-01-01 12:00",
                    "2024-01-01 15:00",
                ]
            ),
        )

    def test_the_deepest_fall_is_still_measured(
        self, intraday_curve, trades, ohlcv_provider
    ):
        metrics = _measure(intraday_curve, trades, ohlcv_provider)

        assert metrics.max_drawdown_amount == pytest.approx(-90.0)

    def test_the_profit_is_still_measured(self, intraday_curve, trades, ohlcv_provider):
        metrics = _measure(intraday_curve, trades, ohlcv_provider)

        assert metrics.net_profit == pytest.approx(30.0)
