import pandas as pd
import pytest

from robottraderslab.analyser import TradeFilter
from robottraderslab.analyser.trade_metrics import compute_trade_metrics

_LONG = TradeFilter(side="long")
_SHORT = TradeFilter(side="short")
_BTC = TradeFilter(symbol="BTC/USDT:USDT")
_HOURS_PER_TRADE = 8


@pytest.fixture
def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [
                "BTC/USDT:USDT",
                "BTC/USDT:USDT",
                "ETH/USDT:USDT",
                "ETH/USDT:USDT",
                "BTC/USDT:USDT",
                "ETH/USDT:USDT",
            ],
            "side": ["long", "short", "long", "short", "long", "long"],
            "entry_time": pd.to_datetime(
                [
                    "2024-01-01 10:00",
                    "2024-01-02 11:00",
                    "2024-01-03 12:00",
                    "2024-01-04 13:00",
                    "2024-01-05 14:00",
                    "2024-01-06 15:00",
                ]
            ),
            "exit_time": pd.to_datetime(
                [
                    "2024-01-01 18:00",
                    "2024-01-02 19:00",
                    "2024-01-03 20:00",
                    "2024-01-04 21:00",
                    "2024-01-05 22:00",
                    "2024-01-06 23:00",
                ]
            ),
            "net_pnl": [100.0, -50.0, 75.0, -25.0, 150.0, 80.0],
            "net_pnl_pct": [0.02, -0.01, 0.015, -0.005, 0.03, 0.016],
            "entry_fee": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "exit_fee": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "entry_reason": [
                "signal_a",
                "signal_b",
                "signal_a",
                "signal_b",
                "signal_a",
                "signal_a",
            ],
            "exit_reason": ["tp", "sl", "tp", "sl", "tp", "tp"],
        }
    )


def _one_sided(side: str, pnls: list[float], exit_reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTC/USDT:USDT"] * len(pnls),
            "side": [side] * len(pnls),
            "entry_time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "exit_time": pd.to_datetime(["2024-01-01 12:00", "2024-01-02 12:00"]),
            "net_pnl": pnls,
            "net_pnl_pct": [pnl / 5000.0 for pnl in pnls],
            "entry_fee": [1.0] * len(pnls),
            "exit_fee": [1.0] * len(pnls),
            "entry_reason": ["signal"] * len(pnls),
            "exit_reason": [exit_reason] * len(pnls),
        }
    )


class TestFilteredTradeMetrics:
    @pytest.mark.parametrize(
        ("trade_filter", "kept"),
        [
            (TradeFilter(), 6),
            (_LONG, 4),
            (_SHORT, 2),
            (_BTC, 3),
            (TradeFilter(side="long", symbol="ETH/USDT:USDT"), 2),
        ],
    )
    def test_a_filter_keeps_its_own_trades(self, trades, trade_filter, kept):
        assert compute_trade_metrics(trades, trade_filter).total_trades == kept

    def test_a_side_is_measured_on_its_own_trades(self, trades):
        long_metrics = compute_trade_metrics(trades, _LONG)

        assert long_metrics.win_rate == 1.0
        assert long_metrics.losing_trades == 0
        assert long_metrics.total_pnl == 405.0

    def test_a_symbol_pays_its_own_fees(self, trades):
        btc_metrics = compute_trade_metrics(trades, _BTC)

        assert btc_metrics.total_fee == 6.0
        assert btc_metrics.biggest_fee == 2.0
        assert btc_metrics.avg_fee == 2.0

    def test_a_side_averages_its_own_results(self, trades):
        assert compute_trade_metrics(trades, _SHORT).avg_trade_pnl == -37.5

    def test_the_kept_trades_alone_state_their_reasons(self, trades):
        long_metrics = compute_trade_metrics(trades, _LONG)

        assert long_metrics.open_reasons.to_dict() == {"signal_a": 4}
        assert long_metrics.close_reasons.to_dict() == {"tp": 4}

    def test_durations_are_averaged_over_the_kept_trades(self, trades):
        long_metrics = compute_trade_metrics(trades, _LONG)

        assert long_metrics.avg_trade_duration_days == pytest.approx(
            _HOURS_PER_TRADE / 24
        )

    def test_streaks_run_over_the_whole_run(self, trades):
        whole_run = compute_trade_metrics(trades)

        assert whole_run.max_win_streak == 2
        assert whole_run.max_lose_streak == 1

    def test_the_risk_reward_ratio_of_the_whole_run(self, trades):
        assert compute_trade_metrics(trades).risk_reward_ratio == pytest.approx(
            101.25 / 37.5
        )

    def test_the_profit_factor_of_the_whole_run(self, trades):
        assert compute_trade_metrics(trades).profit_factor == pytest.approx(
            405.0 / 75.0
        )

    def test_a_run_of_winners_has_no_losses_to_divide_by(self):
        winners = compute_trade_metrics(_one_sided("long", [100.0, 200.0], "tp"))

        assert winners.win_rate == 1.0
        assert winners.profit_factor == float("inf")

    def test_a_run_of_losers_has_no_wins(self):
        losers = compute_trade_metrics(_one_sided("short", [-100.0, -200.0], "sl"))

        assert losers.win_rate == 0.0
        assert losers.profit_factor == 0.0

    def test_the_trades_are_left_untouched(self, trades):
        before = trades.copy()

        compute_trade_metrics(trades, _LONG)

        pd.testing.assert_frame_equal(trades, before)
