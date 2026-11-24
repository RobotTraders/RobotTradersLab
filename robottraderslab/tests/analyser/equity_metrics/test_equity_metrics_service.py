import pandas as pd
import pytest

from robottraderslab.analyser.equity_metrics import compute_equity_metrics


@pytest.fixture
def equity_curve() -> pd.Series:
    return pd.Series(
        [1000 + day * 10 for day in range(90)],
        index=pd.date_range("2024-01-01", periods=90, freq="D"),
        name="equity",
    )


@pytest.fixture
def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2024-01-10", "2024-02-15"]),
            "exit_time": pd.to_datetime(["2024-01-20", "2024-02-25"]),
            "entry_price": [100.0, 110.0],
            "exit_price": [110.0, 120.0],
            "net_quantity": [1.0, 1.0],
            "side": ["long", "long"],
            "symbol": ["BTC/USDT", "BTC/USDT"],
            "net_pnl": [10.0, 10.0],
        }
    )


class TestCalculationMethods:
    def test_the_daily_method_measures_the_curve_itself(
        self, equity_curve, trades, ohlcv_provider
    ):
        metrics = compute_equity_metrics(
            equity_curve, trades, ohlcv_provider, initial_balance=1000.0
        )

        assert metrics.roi == pytest.approx(0.89)
        assert metrics.max_drawdown == 0.0

    def test_the_tradingview_method_measures_the_candles_under_each_trade(
        self, equity_curve, trades, ohlcv_provider
    ):
        ohlcv_provider.get_all_cached_ohlcv_for_symbol.return_value = pd.DataFrame(
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1.0},
            index=pd.date_range("2024-01-01", "2024-12-31", freq="D"),
        )

        metrics = compute_equity_metrics(
            equity_curve,
            trades,
            ohlcv_provider,
            calculation_method="tradingview_like",
            initial_balance=1000.0,
        )

        assert metrics.max_drawdown == pytest.approx(-15.0 / 1010.0)

    def test_an_unknown_method_is_refused(self, equity_curve, trades, ohlcv_provider):
        with pytest.raises(ValueError, match="Unknown calculation method"):
            compute_equity_metrics(
                equity_curve, trades, ohlcv_provider, calculation_method="invalid"
            )


class TestPeriodCovered:
    def test_the_period_spans_the_equity_curve(
        self, equity_curve, trades, ohlcv_provider
    ):
        metrics = compute_equity_metrics(equity_curve, trades, ohlcv_provider)

        assert metrics.period_start == pd.Timestamp("2024-01-01")
        assert metrics.period_end == pd.Timestamp("2024-03-30")

    def test_the_balances_are_the_ends_of_the_equity_curve(
        self, equity_curve, trades, ohlcv_provider
    ):
        metrics = compute_equity_metrics(
            equity_curve, trades, ohlcv_provider, initial_balance=1000.0
        )

        assert metrics.initial_equity == 1000.0
        assert metrics.final_equity == 1890.0

    def test_an_intraday_close_still_ends_the_run(self, trades, ohlcv_provider):
        closing_intraday = pd.Series(
            [1000.0, 1005.0, 1010.0],
            index=pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-02 00:00", "2024-01-02 13:00"]
            ),
        )

        metrics = compute_equity_metrics(
            closing_intraday, trades, ohlcv_provider, initial_balance=1000.0
        )

        assert metrics.period_end == pd.Timestamp("2024-01-02 13:00")
        assert metrics.final_equity == 1010.0


class TestDeepestDrawdown:
    def test_the_amount_comes_from_the_moment_the_share_was_worst(
        self, trades, ohlcv_provider
    ):
        equity = pd.Series(
            [1000.0, 890.0, 2000.0, 1850.0],
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
        )

        metrics = compute_equity_metrics(
            equity, trades, ohlcv_provider, initial_balance=1000.0
        )

        assert metrics.max_drawdown == pytest.approx(-0.11)
        assert metrics.max_drawdown_amount == pytest.approx(-110.0)


class TestReturnOverMaxDrawdown:
    @pytest.mark.parametrize(
        ("equity", "expected"),
        [
            ([1000, 1100, 1050, 1200, 1150, 1300], 6.6),
            ([1000, 1100, 1080, 1200, 1180, 1300, 1280, 1400], 22.0),
            ([1000, 950, 900, 850, 800], -1.0),
            ([1000, 1100, 800, 1200], 0.733),
        ],
    )
    def test_the_return_is_measured_per_unit_of_the_deepest_fall(
        self, trades, ohlcv_provider, equity, expected
    ):
        curve = pd.Series(
            equity, index=pd.date_range("2024-01-01", periods=len(equity), freq="D")
        )

        metrics = compute_equity_metrics(
            curve, trades, ohlcv_provider, initial_balance=1000.0
        )

        assert metrics.return_over_max_drawdown == pytest.approx(expected, rel=1e-2)

    def test_a_curve_that_never_fell_has_no_drawdown_to_divide_by(
        self, trades, ohlcv_provider
    ):
        flat = pd.Series(
            [1000.0] * 4, index=pd.date_range("2024-01-01", periods=4, freq="D")
        )

        metrics = compute_equity_metrics(
            flat, trades, ohlcv_provider, initial_balance=1000.0
        )

        assert metrics.return_over_max_drawdown == 0.0
