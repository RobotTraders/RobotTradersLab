import pandas as pd
import pytest

from robottraderslab.analyser.equity_metrics.equity_metrics import (
    calculate_sortino_ratio,
)
from robottraderslab.analyser.equity_metrics.tradingview_like import (
    calculate_tradingview_drawdowns,
    calculate_tradingview_sortino_ratio,
    filter_equity_at_trade_exits,
    filter_equity_at_trade_times,
    resample_to_monthly,
)

_INITIAL_CAPITAL = 10000.0
_QUANTITY = 10.0
_ENTRY_PRICE = 100.0
_MONTHLY_RISK_FREE_RATE = 0.02 / 12.0
_MIXED_RETURNS = pd.Series([0.05, -0.02, 0.03, -0.01, 0.04, 0.02])


@pytest.fixture
def daily_equity() -> pd.Series:
    return pd.Series(
        [10000 + day * 100 for day in range(90)],
        index=pd.date_range("2024-01-01", periods=90, freq="D"),
    )


@pytest.fixture
def two_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2024-01-10", "2024-01-20"]),
            "exit_time": pd.to_datetime(["2024-01-15", "2024-01-25"]),
        }
    )


@pytest.fixture
def five_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": [105.0, 104.0, 106.0, 103.0, 107.0],
            "low": [95.0, 96.0, 94.0, 97.0, 93.0],
            "close": 102.0,
            "volume": 1000.0,
        },
        index=pd.date_range("2024-01-10", periods=5, freq="D"),
    )


@pytest.fixture
def ten_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": [94.0, 95.0, 96.0, 97.0, 98.0, 96.0, 97.0, 98.0, 99.0, 99.0],
            "close": 100.0,
            "volume": 1000.0,
        },
        index=pd.date_range("2024-01-10", periods=10, freq="D"),
    )


def _one_trade(side: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": [pd.Timestamp("2024-01-10")],
            "exit_time": [pd.Timestamp("2024-01-14")],
            "entry_price": [_ENTRY_PRICE],
            "exit_price": [102.0],
            "net_quantity": [_QUANTITY],
            "side": [side],
            "symbol": ["BTC/USDT"],
            "net_pnl": [20.0],
        }
    )


def _two_long_trades(first_pnl: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2024-01-10", "2024-01-15"]),
            "exit_time": pd.to_datetime(["2024-01-14", "2024-01-19"]),
            "entry_price": [_ENTRY_PRICE, _ENTRY_PRICE],
            "exit_price": [_ENTRY_PRICE + first_pnl / _QUANTITY, 101.0],
            "net_quantity": [_QUANTITY, _QUANTITY],
            "side": ["long", "long"],
            "symbol": ["BTC/USDT", "BTC/USDT"],
            "net_pnl": [first_pnl, 10.0],
        }
    )


class TestResampleToMonthly:
    def test_each_month_ends_on_its_last_reading(self, daily_equity):
        monthly = resample_to_monthly(daily_equity)

        assert monthly.index.tolist() == [
            pd.Timestamp("2023-12-01"),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
            pd.Timestamp("2024-03-01"),
        ]
        assert monthly.tolist() == [
            daily_equity["2024-01-01"],
            daily_equity["2024-02-01"],
            daily_equity["2024-03-01"],
            daily_equity["2024-03-30"],
        ]

    def test_a_reading_on_the_first_midnight_ends_the_month_before(self):
        equity = pd.Series(
            [100.0, 110.0, 120.0],
            index=pd.to_datetime(
                ["2024-01-31 23:00", "2024-02-01 00:00", "2024-02-01 01:00"]
            ),
        )

        monthly = resample_to_monthly(equity)

        assert monthly.to_dict() == {
            pd.Timestamp("2024-01-01"): 110.0,
            pd.Timestamp("2024-02-01"): 120.0,
        }

    def test_a_month_without_readings_is_left_out(self):
        january_and_march = pd.Series(
            [100.0, 110.0], index=pd.to_datetime(["2024-01-31", "2024-03-31"])
        )

        monthly = resample_to_monthly(january_and_march)

        assert monthly.index.tolist() == [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-03-01"),
        ]


class TestFilterEquityAtTradeTimes:
    def test_the_curve_is_read_at_its_ends_and_at_every_entry_and_exit(
        self, daily_equity, two_trades
    ):
        sampled = filter_equity_at_trade_times(daily_equity, two_trades)

        assert sampled.index.tolist() == [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-10"),
            pd.Timestamp("2024-01-15"),
            pd.Timestamp("2024-01-20"),
            pd.Timestamp("2024-01-25"),
            pd.Timestamp("2024-03-30"),
        ]


class TestFilterEquityAtTradeExits:
    def test_the_curve_is_read_at_its_ends_and_at_every_exit(
        self, daily_equity, two_trades
    ):
        sampled = filter_equity_at_trade_exits(daily_equity, two_trades)

        assert sampled.index.tolist() == [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-15"),
            pd.Timestamp("2024-01-25"),
            pd.Timestamp("2024-03-30"),
        ]

    def test_an_open_trade_has_no_exit_to_read_at(self, daily_equity):
        one_still_open = pd.DataFrame(
            {
                "entry_time": pd.to_datetime(["2024-01-10", "2024-01-20"]),
                "exit_time": pd.to_datetime(["2024-01-15", None]),
            }
        )

        sampled = filter_equity_at_trade_exits(daily_equity, one_still_open)

        assert sampled.index.tolist() == [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-15"),
            pd.Timestamp("2024-03-30"),
        ]


class TestCalculateTradingviewDrawdowns:
    def test_a_run_without_trades_has_no_drawdown(self, ohlcv_provider):
        drawdowns = calculate_tradingview_drawdowns(
            _one_trade("long").iloc[0:0], _INITIAL_CAPITAL, ohlcv_provider
        )

        assert drawdowns.absolute_drawdown.empty
        assert drawdowns.max_drawdown == 0.0
        assert drawdowns.max_drawdown_amount == 0.0

    def test_a_long_trade_draws_down_to_the_lowest_low(
        self, ohlcv_provider, five_candles
    ):
        ohlcv_provider.get_all_cached_ohlcv_for_symbol.return_value = five_candles

        drawdowns = calculate_tradingview_drawdowns(
            _one_trade("long"), _INITIAL_CAPITAL, ohlcv_provider
        )

        assert drawdowns.max_drawdown == pytest.approx(
            -_QUANTITY * (_ENTRY_PRICE - 93.0) / _INITIAL_CAPITAL
        )

    def test_a_short_trade_draws_down_to_the_highest_high(
        self, ohlcv_provider, five_candles
    ):
        ohlcv_provider.get_all_cached_ohlcv_for_symbol.return_value = five_candles

        drawdowns = calculate_tradingview_drawdowns(
            _one_trade("short"), _INITIAL_CAPITAL, ohlcv_provider
        )

        assert drawdowns.max_drawdown == pytest.approx(
            -_QUANTITY * (107.0 - _ENTRY_PRICE) / _INITIAL_CAPITAL
        )

    def test_every_candle_under_a_trade_gets_a_reading(
        self, ohlcv_provider, five_candles
    ):
        ohlcv_provider.get_all_cached_ohlcv_for_symbol.return_value = five_candles

        drawdowns = calculate_tradingview_drawdowns(
            _one_trade("long"), _INITIAL_CAPITAL, ohlcv_provider
        )

        assert drawdowns.absolute_drawdown.index.tolist() == five_candles.index.tolist()

    def test_a_loss_deepens_the_next_trade_by_the_distance_to_the_peak(
        self, ohlcv_provider, ten_candles
    ):
        ohlcv_provider.get_all_cached_ohlcv_for_symbol.return_value = ten_candles

        drawdowns = calculate_tradingview_drawdowns(
            _two_long_trades(first_pnl=-30.0), _INITIAL_CAPITAL, ohlcv_provider
        )

        assert drawdowns.max_drawdown_amount == pytest.approx(-(30.0 + 40.0))

    def test_a_share_divides_by_the_peak_equity_at_the_time(
        self, ohlcv_provider, ten_candles
    ):
        ohlcv_provider.get_all_cached_ohlcv_for_symbol.return_value = ten_candles

        drawdowns = calculate_tradingview_drawdowns(
            _two_long_trades(first_pnl=50.0), _INITIAL_CAPITAL, ohlcv_provider
        )

        assert drawdowns.drawdown_percentage["2024-01-15"] == pytest.approx(
            -40.0 / (_INITIAL_CAPITAL + 50.0)
        )


class TestCalculateTradingViewSortinoRatio:
    def test_mixed_returns(self):
        sortino = calculate_tradingview_sortino_ratio(
            _MIXED_RETURNS,
            annualization_factor=0.0,
            risk_free_rate=_MONTHLY_RISK_FREE_RATE,
        )

        assert sortino == pytest.approx(1.659, rel=1e-2)

    def test_agrees_with_the_traditional_sortino(self):
        tradingview = calculate_tradingview_sortino_ratio(
            _MIXED_RETURNS, 0.0, _MONTHLY_RISK_FREE_RATE
        )
        traditional = calculate_sortino_ratio(
            _MIXED_RETURNS, 0.0, _MONTHLY_RISK_FREE_RATE
        )

        assert tradingview == pytest.approx(traditional, rel=1e-6)

    def test_without_a_losing_period_there_is_no_downside_to_measure(self):
        only_gains = pd.Series([0.02, 0.03, 0.01, 0.02, 0.025])

        sortino = calculate_tradingview_sortino_ratio(
            only_gains, annualization_factor=0.0, risk_free_rate=0.0
        )

        assert sortino == 0.0

    def test_annualisation_scales_by_the_root_of_the_periods(self):
        per_period = calculate_tradingview_sortino_ratio(
            _MIXED_RETURNS, annualization_factor=0.0, risk_free_rate=0.0
        )

        annualised = calculate_tradingview_sortino_ratio(
            _MIXED_RETURNS, annualization_factor=12.0, risk_free_rate=0.0
        )

        assert annualised == pytest.approx(per_period * (12.0**0.5), rel=1e-6)
