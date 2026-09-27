import math

import numpy as np
import pandas as pd
import pytest

from robottraderslab.analyser.trade_metrics.trade_metrics import (
    calculate_average_pnl_metrics,
    calculate_fee_metrics,
    calculate_largest_trades,
    calculate_profit_factor,
    calculate_reason_metrics,
    calculate_risk_reward_ratio,
    calculate_streaks,
    calculate_time_in_position_metrics,
    calculate_trade_based_max_drawdown,
    calculate_trade_counts,
    calculate_trade_duration_metrics,
    calculate_win_rate,
)
from robottraderslab.analyser.trade_metrics.trade_metrics_models import (
    AveragePnlMetrics,
)


@pytest.fixture
def pnl_series() -> pd.Series:
    return pd.Series(
        [100, -50, 150, -40, 130, -38, -80, -43, 136, 120, -14, 51],
        name="pnl",
    )


@pytest.fixture
def fee_series() -> pd.Series:
    return pd.Series(
        [1.5, 2.0, 1.0, 2.5, 1.8, 1.6, 3.0, 1.2, 1.9, 2.2, 1.4, 1.7],
        name="fees",
    )


@pytest.fixture
def trades_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                [
                    "2023-01-01 10:00:00",
                    "2023-01-02 14:00:00",
                    "2023-01-03 09:00:00",
                    "2023-01-04 16:00:00",
                ]
            ),
            "exit_time": pd.to_datetime(
                [
                    "2023-01-01 18:00:00",
                    "2023-01-04 14:00:00",
                    "2023-01-05 09:00:00",
                    "2023-01-06 16:00:00",
                ]
            ),
            "net_pnl": [100, -50, 150, -40],
        }
    )


class TestCalculateWinRate:
    @pytest.mark.parametrize(
        ("pnl_values", "expected_rate"),
        [
            ([100, 100, 100], 1.0),
            ([-100, -100, -100], 0.0),
            ([100, -100, 100], 0.6667),
        ],
    )
    def test_with_various_win_patterns(self, pnl_values, expected_rate):
        pnl = pd.Series(pnl_values)
        win_rate = calculate_win_rate(pnl)
        assert win_rate == pytest.approx(expected_rate, rel=1e-3)

    def test_with_empty_series(self):
        pnl = pd.Series([], dtype=float)
        win_rate = calculate_win_rate(pnl)
        assert win_rate == 0.0


class TestCalculateProfitFactor:
    @pytest.mark.parametrize(
        ("pnl_values", "expected_pf"),
        [
            ([100, 100, -50], 4.0),
            ([50, -100, -50], 0.3333),
        ],
    )
    def test_with_mixed_trades(self, pnl_values, expected_pf):
        pnl = pd.Series(pnl_values)
        profit_factor = calculate_profit_factor(pnl)
        assert profit_factor == pytest.approx(expected_pf, rel=1e-3)

    def test_with_only_winning_trades(self):
        pnl = pd.Series([100, 100, 100])
        profit_factor = calculate_profit_factor(pnl)
        assert np.isinf(profit_factor)

    def test_with_empty_series(self):
        pnl = pd.Series([], dtype=float)
        profit_factor = calculate_profit_factor(pnl)
        assert np.isinf(profit_factor)


class TestCalculateStreaks:
    @pytest.mark.parametrize(
        ("pnl_values", "expected_win_streak", "expected_lose_streak"),
        [
            ([100, 100, 100, -100], 3, 1),
            ([-100, -100, -100, 100], 1, 3),
            ([100, -100, 100, -100, 100], 1, 1),
            ([100, 100, -100, -100, -100, 100, 100, 100, 100], 4, 3),
        ],
    )
    def test_with_various_streak_patterns(
        self, pnl_values, expected_win_streak, expected_lose_streak
    ):
        pnl = pd.Series(pnl_values)
        streaks = calculate_streaks(pnl)
        assert streaks.max_win_streak == expected_win_streak
        assert streaks.max_lose_streak == expected_lose_streak

    def test_with_empty_series(self):
        pnl = pd.Series([], dtype=float)
        streaks = calculate_streaks(pnl)
        assert streaks.max_win_streak == 0
        assert streaks.max_lose_streak == 0


class TestCalculateFeeMetrics:
    def test_with_data(self, fee_series):
        fees = calculate_fee_metrics(fee_series)
        assert fees.total_fee == pytest.approx(21.8, rel=1e-3)
        assert fees.biggest_fee == pytest.approx(3.0, rel=1e-3)
        assert fees.avg_fee == pytest.approx(1.816667, rel=1e-3)

    def test_with_empty_series(self):
        empty_fees = pd.Series([], dtype=float)
        fees = calculate_fee_metrics(empty_fees)
        assert fees.total_fee == 0.0
        assert np.isnan(fees.biggest_fee)
        assert np.isnan(fees.avg_fee)


class TestCalculateTradeCounts:
    @pytest.mark.parametrize(
        ("pnl_values", "expected_total", "expected_winning", "expected_losing"),
        [
            ([100, -50, 150], 3, 2, 1),
            ([100, 100, 100], 3, 3, 0),
            ([-50, -50, -50], 3, 0, 3),
        ],
    )
    def test_with_various_trade_patterns(
        self, pnl_values, expected_total, expected_winning, expected_losing
    ):
        pnl = pd.Series(pnl_values)
        counts = calculate_trade_counts(pnl)
        assert counts.total_trades == expected_total
        assert counts.winning_trades == expected_winning
        assert counts.losing_trades == expected_losing

    def test_with_empty_series(self):
        pnl = pd.Series([], dtype=float)
        counts = calculate_trade_counts(pnl)
        assert counts.total_trades == 0
        assert counts.winning_trades == 0
        assert counts.losing_trades == 0


class TestCalculateAveragePnlMetrics:
    def test_with_mixed_trades(self):
        pnl = pd.Series([100, -50, 150, -40])
        averages = calculate_average_pnl_metrics(pnl, pd.DataFrame())
        assert averages.avg_trade_pnl == pytest.approx(40.0, rel=1e-3)
        assert averages.avg_winning_trade_pnl == pytest.approx(125.0, rel=1e-3)
        assert averages.avg_losing_trade_pnl == pytest.approx(45.0, rel=1e-3)
        assert averages.avg_trade_return == 0.0
        assert averages.avg_winning_trade_return == 0.0
        assert averages.avg_losing_trade_return == 0.0

    def test_with_empty_series(self):
        empty_pnl = pd.Series([], dtype=float)
        averages = calculate_average_pnl_metrics(empty_pnl, pd.DataFrame())
        assert averages.avg_trade_pnl == 0.0
        assert averages.avg_winning_trade_pnl == 0.0
        assert averages.avg_losing_trade_pnl == 0.0
        assert averages.avg_trade_return == 0.0
        assert averages.avg_winning_trade_return == 0.0
        assert averages.avg_losing_trade_return == 0.0

    def test_with_only_winning_trades(self):
        pnl = pd.Series([100, 150, 200])
        averages = calculate_average_pnl_metrics(pnl, pd.DataFrame())
        assert averages.avg_trade_pnl == pytest.approx(150.0, rel=1e-3)
        assert averages.avg_winning_trade_pnl == pytest.approx(150.0, rel=1e-3)
        assert averages.avg_losing_trade_pnl == 0.0
        assert averages.avg_trade_return == 0.0
        assert averages.avg_winning_trade_return == 0.0
        assert averages.avg_losing_trade_return == 0.0

    def test_with_only_losing_trades(self):
        pnl = pd.Series([-50, -100, -75])
        averages = calculate_average_pnl_metrics(pnl, pd.DataFrame())
        assert averages.avg_trade_pnl == pytest.approx(-75.0, rel=1e-3)
        assert averages.avg_winning_trade_pnl == 0.0
        assert averages.avg_losing_trade_pnl == pytest.approx(75.0, rel=1e-3)
        assert averages.avg_trade_return == 0.0
        assert averages.avg_winning_trade_return == 0.0
        assert averages.avg_losing_trade_return == 0.0

    def test_with_percentage_data(self):
        pnl = pd.Series([100, -50, 150, -40])
        trades_df = pd.DataFrame(
            {
                "net_pnl": [100, -50, 150, -40],
                "net_pnl_pct": [0.01, -0.005, 0.015, -0.004],
            }
        )
        averages = calculate_average_pnl_metrics(pnl, trades_df)
        assert averages.avg_trade_pnl == pytest.approx(40.0, rel=1e-3)
        assert averages.avg_winning_trade_pnl == pytest.approx(125.0, rel=1e-3)
        assert averages.avg_losing_trade_pnl == pytest.approx(45.0, rel=1e-3)
        assert averages.avg_trade_return == pytest.approx(0.4, rel=1e-3)
        assert averages.avg_winning_trade_return == pytest.approx(1.25, rel=1e-3)
        assert averages.avg_losing_trade_return == pytest.approx(-0.45, rel=1e-3)


class TestCalculateRiskRewardRatio:
    @pytest.mark.parametrize(
        ("win_avg_pct", "lose_avg_pct", "expected_ratio"),
        [
            (100.0, -50.0, 2.0),
            (175.0, -75.0, 2.3333),
        ],
    )
    def test_with_avg_percent_inputs(self, win_avg_pct, lose_avg_pct, expected_ratio):
        avg = AveragePnlMetrics(
            avg_trade_pnl=0.0,
            avg_winning_trade_pnl=0.0,
            avg_losing_trade_pnl=0.0,
            avg_trade_return=0.0,
            avg_winning_trade_return=win_avg_pct,
            avg_losing_trade_return=lose_avg_pct,
        )
        risk_reward = calculate_risk_reward_ratio(avg)
        assert risk_reward == pytest.approx(expected_ratio, rel=1e-3)

    def test_with_only_winning_trades(self):
        avg = AveragePnlMetrics(
            avg_trade_pnl=0.0,
            avg_winning_trade_pnl=0.0,
            avg_losing_trade_pnl=0.0,
            avg_trade_return=0.0,
            avg_winning_trade_return=1.0,
            avg_losing_trade_return=0.0,
        )
        risk_reward = calculate_risk_reward_ratio(avg)
        assert np.isinf(risk_reward)

    def test_with_only_losing_trades(self):
        avg = AveragePnlMetrics(
            avg_trade_pnl=0.0,
            avg_winning_trade_pnl=0.0,
            avg_losing_trade_pnl=0.0,
            avg_trade_return=0.0,
            avg_winning_trade_return=0.0,
            avg_losing_trade_return=-1.0,
        )
        risk_reward = calculate_risk_reward_ratio(avg)
        assert risk_reward == 0.0

    def test_with_no_trades(self):
        avg = AveragePnlMetrics(
            avg_trade_pnl=0.0,
            avg_winning_trade_pnl=0.0,
            avg_losing_trade_pnl=0.0,
            avg_trade_return=0.0,
            avg_winning_trade_return=0.0,
            avg_losing_trade_return=0.0,
        )
        risk_reward = calculate_risk_reward_ratio(avg)
        assert risk_reward == 0.0


class TestCalculateLargestTrades:
    def test_with_mixed_trades(self):
        pnl = pd.Series([100, -50, 250, -120, 80])
        largest = calculate_largest_trades(pnl, pnl / 1000.0)
        assert largest.largest_winning_trade_pnl == pytest.approx(250.0, rel=1e-3)
        assert largest.largest_losing_trade_pnl == pytest.approx(120.0, rel=1e-3)

    def test_with_empty_series(self):
        empty_pnl = pd.Series([], dtype=float)
        largest = calculate_largest_trades(empty_pnl, empty_pnl)
        assert largest.largest_winning_trade_pnl == 0.0
        assert largest.largest_losing_trade_pnl == 0.0

    def test_with_only_winning_trades(self):
        pnl = pd.Series([100, 150, 200])
        largest = calculate_largest_trades(pnl, pnl / 1000.0)
        assert largest.largest_winning_trade_pnl == pytest.approx(200.0, rel=1e-3)
        assert largest.largest_losing_trade_pnl == 0.0

    def test_with_only_losing_trades(self):
        pnl = pd.Series([-50, -100, -75])
        largest = calculate_largest_trades(pnl, pnl / 1000.0)
        assert largest.largest_winning_trade_pnl == 0.0
        assert largest.largest_losing_trade_pnl == pytest.approx(100.0, rel=1e-3)


class TestBestAndWorstTradeReturns:
    def test_the_best_return_is_the_largest_share_made(self):
        largest = calculate_largest_trades(
            pd.Series([100.0, -50.0, 30.0]), pd.Series([0.02, -0.05, 0.03])
        )

        assert largest.best_trade_return == 0.03

    def test_the_worst_return_keeps_its_sign(self):
        largest = calculate_largest_trades(
            pd.Series([100.0, -50.0, 30.0]), pd.Series([0.02, -0.05, 0.03])
        )

        assert largest.worst_trade_return == -0.05

    def test_a_run_without_trades_has_no_returns(self):
        no_trades = pd.Series(dtype=float)

        largest = calculate_largest_trades(no_trades, no_trades)

        assert largest.best_trade_return == 0.0
        assert largest.worst_trade_return == 0.0


class TestCalculateTradeDurationMetrics:
    def test_with_mixed_trades(self, trades_df):
        durations = calculate_trade_duration_metrics(trades_df)
        assert durations.avg_trade_duration_days == pytest.approx(1.583, rel=1e-2)
        assert durations.avg_winning_trade_duration_days == pytest.approx(
            1.167, rel=1e-2
        )
        assert durations.avg_losing_trade_duration_days == pytest.approx(2.0, rel=1e-3)

    def test_with_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=["entry_time", "exit_time", "net_pnl"])
        durations = calculate_trade_duration_metrics(empty_df)
        assert durations.avg_trade_duration_days == 0.0
        assert durations.avg_winning_trade_duration_days == 0.0
        assert durations.avg_losing_trade_duration_days == 0.0

    def test_with_only_winning_trades(self):
        trades = pd.DataFrame(
            {
                "entry_time": pd.to_datetime(
                    ["2023-01-01 10:00:00", "2023-01-02 10:00:00"]
                ),
                "exit_time": pd.to_datetime(
                    ["2023-01-02 10:00:00", "2023-01-04 10:00:00"]
                ),
                "net_pnl": [100, 150],
            }
        )
        durations = calculate_trade_duration_metrics(trades)
        assert durations.avg_trade_duration_days == pytest.approx(1.5, rel=1e-3)
        assert durations.avg_winning_trade_duration_days == pytest.approx(1.5, rel=1e-3)
        assert durations.avg_losing_trade_duration_days == 0.0

    def test_with_only_losing_trades(self):
        trades = pd.DataFrame(
            {
                "entry_time": pd.to_datetime(
                    ["2023-01-01 10:00:00", "2023-01-02 10:00:00"]
                ),
                "exit_time": pd.to_datetime(
                    ["2023-01-02 10:00:00", "2023-01-04 10:00:00"]
                ),
                "net_pnl": [-50, -100],
            }
        )
        durations = calculate_trade_duration_metrics(trades)
        assert durations.avg_trade_duration_days == pytest.approx(1.5, rel=1e-3)
        assert durations.avg_winning_trade_duration_days == 0.0
        assert durations.avg_losing_trade_duration_days == pytest.approx(1.5, rel=1e-3)


class TestCalculateTimeInPositionMetrics:
    def test_with_trades_data(self):
        trades_df = pd.DataFrame(
            {
                "entry_time": pd.to_datetime(
                    [
                        "2024-01-01 09:00:00",
                        "2024-01-02 10:00:00",
                        "2024-01-04 14:00:00",
                    ]
                ),
                "exit_time": pd.to_datetime(
                    [
                        "2024-01-01 17:00:00",
                        "2024-01-03 10:00:00",
                        "2024-01-05 14:00:00",
                    ]
                ),
                "net_pnl": [100, -50, 75],
            }
        )
        time_in_position = calculate_time_in_position_metrics(trades_df)
        assert time_in_position.time_in_position_ratio == pytest.approx(0.554, rel=1e-2)
        assert time_in_position.avg_trades_per_day == pytest.approx(0.6, rel=1e-3)

    def test_with_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=["entry_time", "exit_time"])
        time_in_position = calculate_time_in_position_metrics(empty_df)
        assert time_in_position.time_in_position_ratio == 0.0
        assert time_in_position.avg_trades_per_day == 0.0

    def test_with_single_trade(self):
        trades_df = pd.DataFrame(
            {
                "entry_time": pd.to_datetime(["2024-01-01 09:00:00"]),
                "exit_time": pd.to_datetime(["2024-01-01 17:00:00"]),
                "net_pnl": [100],
            }
        )
        time_in_position = calculate_time_in_position_metrics(trades_df)
        assert time_in_position.time_in_position_ratio == pytest.approx(1.0, rel=1e-3)
        assert time_in_position.avg_trades_per_day == pytest.approx(1.0, rel=1e-3)

    def test_with_overlapping_trades_same_day(self):
        trades_df = pd.DataFrame(
            {
                "entry_time": pd.to_datetime(
                    ["2024-01-01 09:00:00", "2024-01-01 11:00:00"]
                ),
                "exit_time": pd.to_datetime(
                    [
                        "2024-01-01 10:00:00",
                        "2024-01-01 13:00:00",
                    ]
                ),
                "net_pnl": [50, 75],
            }
        )
        time_in_position = calculate_time_in_position_metrics(trades_df)
        assert time_in_position.time_in_position_ratio == pytest.approx(0.75, rel=1e-3)
        assert time_in_position.avg_trades_per_day == pytest.approx(2.0, rel=1e-3)


class TestCalculateReasonMetrics:
    def test_with_entry_exit_reason_columns(self):
        trades_df = pd.DataFrame(
            {
                "entry_reason": ["signal", "signal", "manual", "signal"],
                "exit_reason": ["take_profit", "stop_loss", "take_profit", "signal"],
                "net_pnl": [100, -50, 75, -25],
            }
        )
        reasons = calculate_reason_metrics(trades_df)
        expected_open = pd.Series({"signal": 3, "manual": 1}, name="entry_reason")
        expected_close = pd.Series(
            {"take_profit": 2, "stop_loss": 1, "signal": 1}, name="exit_reason"
        )
        pd.testing.assert_series_equal(reasons.open_reasons, expected_open)
        pd.testing.assert_series_equal(reasons.close_reasons, expected_close)

    def test_with_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=["entry_reason", "exit_reason", "net_pnl"])
        reasons = calculate_reason_metrics(empty_df)
        assert reasons.open_reasons.empty
        assert reasons.close_reasons.empty

    def test_with_single_reason_type(self):
        trades_df = pd.DataFrame(
            {
                "entry_reason": ["signal"] * 5,
                "exit_reason": ["take_profit"] * 5,
                "net_pnl": [100, -50, 75, -25, 30],
            }
        )
        reasons = calculate_reason_metrics(trades_df)
        expected_open = pd.Series({"signal": 5}, name="entry_reason")
        expected_close = pd.Series({"take_profit": 5}, name="exit_reason")
        pd.testing.assert_series_equal(reasons.open_reasons, expected_open)
        pd.testing.assert_series_equal(reasons.close_reasons, expected_close)


class TestCalculateTradeBasedMaxDrawdown:
    def test_with_normal_drawdown_sequence(self):
        equity = pd.Series(
            [10000, 10500, 10200, 10800, 10300, 11000],
            index=pd.date_range("2024-01-01", periods=6, freq="D"),
        )
        trades = pd.DataFrame(
            {
                "exit_time": pd.to_datetime(
                    [
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-04",
                        "2024-01-05",
                        "2024-01-06",
                    ]
                )
            }
        )

        drawdown = calculate_trade_based_max_drawdown(equity, trades)

        assert drawdown.max_drawdown == pytest.approx(-0.0463, rel=1e-3)

    def test_with_continuous_growth_no_drawdown(self):
        equity = pd.Series(
            [10000, 10500, 11000, 11500, 12000],
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        )
        trades = pd.DataFrame(
            {"exit_time": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])}
        )

        drawdown = calculate_trade_based_max_drawdown(equity, trades)

        assert drawdown.max_drawdown == 0.0

    def test_with_empty_trades(self):
        equity = pd.Series(
            [10000, 10500, 11000],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        trades = pd.DataFrame({"exit_time": pd.Series([], dtype="datetime64[ns]")})

        drawdown = calculate_trade_based_max_drawdown(equity, trades)

        assert drawdown.max_drawdown == 0.0

    def test_with_empty_equity(self):
        equity = pd.Series([], dtype=float)
        trades = pd.DataFrame({"exit_time": pd.to_datetime(["2024-01-02"])})

        drawdown = calculate_trade_based_max_drawdown(equity, trades)

        assert drawdown.max_drawdown == 0.0

    def test_samples_only_at_trade_exits(self):
        equity = pd.Series(
            [10000, 10500, 10800, 10200, 10600, 10100, 11000],
            index=pd.date_range("2024-01-01", periods=7, freq="D"),
        )
        trades = pd.DataFrame(
            {"exit_time": pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-07"])}
        )

        drawdown = calculate_trade_based_max_drawdown(equity, trades)

        assert drawdown.max_drawdown == pytest.approx(-0.0286, rel=1e-3)

    def test_includes_starting_equity(self):
        equity = pd.Series(
            [10000, 9500, 10500],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        trades = pd.DataFrame(
            {"exit_time": pd.to_datetime(["2024-01-02", "2024-01-03"])}
        )

        drawdown = calculate_trade_based_max_drawdown(equity, trades)

        assert drawdown.max_drawdown == pytest.approx(-0.05, rel=1e-6)

    def test_the_currency_the_deepest_fall_cost(self):
        equity = pd.Series(
            [10000, 10500, 10200],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        trades = pd.DataFrame(
            {"exit_time": pd.to_datetime(["2024-01-02", "2024-01-03"])}
        )

        drawdown = calculate_trade_based_max_drawdown(equity, trades)

        assert drawdown.amount == pytest.approx(-300.0)

    def test_a_curve_with_nothing_to_measure_states_no_amount(self):
        equity = pd.Series([], dtype=float)
        trades = pd.DataFrame({"exit_time": pd.to_datetime(["2024-01-02"])})

        drawdown = calculate_trade_based_max_drawdown(equity, trades)

        assert drawdown.amount is None


class TestACurveThatPeaksAtZero:
    @pytest.fixture
    def trades(self) -> pd.DataFrame:
        return pd.DataFrame({"exit_time": pd.to_datetime(["2024-01-02", "2024-01-03"])})

    @pytest.fixture
    def losing_from_the_first_trade(self) -> pd.Series:
        return pd.Series(
            [0.0, -40.0, -90.0],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

    def test_a_loss_before_any_gain_states_a_finite_share(
        self, losing_from_the_first_trade, trades
    ):
        drawdown = calculate_trade_based_max_drawdown(
            losing_from_the_first_trade, trades
        )

        assert math.isfinite(drawdown.max_drawdown)

    def test_a_loss_before_any_gain_states_no_amount(
        self, losing_from_the_first_trade, trades
    ):
        drawdown = calculate_trade_based_max_drawdown(
            losing_from_the_first_trade, trades
        )

        assert drawdown.amount is None

    def test_a_gain_after_the_zero_peak_is_measured_from_where_it_reached(self, trades):
        equity = pd.Series(
            [0.0, 100.0, 60.0],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        drawdown = calculate_trade_based_max_drawdown(equity, trades)

        assert drawdown.max_drawdown == pytest.approx(-0.4)
        assert drawdown.amount == pytest.approx(-40.0)
