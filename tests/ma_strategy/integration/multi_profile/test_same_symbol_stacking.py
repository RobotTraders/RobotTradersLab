from pathlib import Path

import pytest

from robottraderslab.bootstrap import BotConfig

TOML_CONFIG = """\
[backtest]
initial_balance = { USDT = 10000.0 }
maker_fee_rate = 0.001
taker_fee_rate = 0.001
fee_mode = "cost"
start_date = "2024-01-01"
end_date = "2024-12-31"
verbose = false

[backtest.ohlcv_provider]
ohlcv_provider = "csv"
file = "../single_symbol_pipeline/data/btc_usdt_binance_2024.csv"
symbol = "BTC/USDT:USDT"
timeframe = "1d"

[strategy]
strategy_class = "futures_ma"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1d"
available_balance_ratio = 0.25
fast_ma_length = 3
slow_ma_length = 10
trend_ma_length = 40
tag = "fast"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1d"
available_balance_ratio = 0.25
fast_ma_length = 7
slow_ma_length = 15
trend_ma_length = 40
tag = "slow"
"""


@pytest.fixture(scope="class")
def bot_config() -> BotConfig:
    bot_config = BotConfig.from_text(TOML_CONFIG)
    assert bot_config.backtest is not None
    csv_file = Path(__file__).parent / bot_config.backtest.ohlcv_provider["file"]
    bot_config.backtest.ohlcv_provider["file"] = str(csv_file)
    return bot_config


class TestSameSymbolStacking:
    """Two trend-filtered profiles on BTC@1d produce stacked positions on both sides."""

    EXPECTED_FILL_COUNTS = {
        "enter_long": 7,
        "add_to_long": 8,
        "reduce_long": 8,
        "exit_long": 7,
        "enter_short": 5,
        "add_to_short": 4,
        "reduce_short": 4,
        "exit_short": 5,
    }
    EXPECTED_CLOSED_TRADES = 28
    EXPECTED_FINAL_BALANCE = 9836.526783776962
    EXPECTED_MAX_DRAWDOWN = -0.20245511166962907
    TOLERANCE = 0.01

    def test_both_profiles_produce_fills(self, backtest_outputs):
        tags = set(backtest_outputs.fills["tag"].unique())

        assert tags == {"1d-fast", "1d-slow"}

    def test_fills_only_on_btc(self, backtest_outputs):
        symbols = set(backtest_outputs.fills["symbol"].unique())

        assert symbols == {"BTC/USDT:USDT"}

    def test_enter_long_count(self, backtest_outputs):
        count = (backtest_outputs.fills["fill_type"] == "enter_long").sum()

        assert count == self.EXPECTED_FILL_COUNTS["enter_long"]

    def test_add_to_long_count(self, backtest_outputs):
        count = (backtest_outputs.fills["fill_type"] == "add_to_long").sum()

        assert count == self.EXPECTED_FILL_COUNTS["add_to_long"]

    def test_reduce_long_count(self, backtest_outputs):
        count = (backtest_outputs.fills["fill_type"] == "reduce_long").sum()

        assert count == self.EXPECTED_FILL_COUNTS["reduce_long"]

    def test_exit_long_count(self, backtest_outputs):
        count = (backtest_outputs.fills["fill_type"] == "exit_long").sum()

        assert count == self.EXPECTED_FILL_COUNTS["exit_long"]

    def test_enter_short_count(self, backtest_outputs):
        count = (backtest_outputs.fills["fill_type"] == "enter_short").sum()

        assert count == self.EXPECTED_FILL_COUNTS["enter_short"]

    def test_add_to_short_count(self, backtest_outputs):
        count = (backtest_outputs.fills["fill_type"] == "add_to_short").sum()

        assert count == self.EXPECTED_FILL_COUNTS["add_to_short"]

    def test_reduce_short_count(self, backtest_outputs):
        count = (backtest_outputs.fills["fill_type"] == "reduce_short").sum()

        assert count == self.EXPECTED_FILL_COUNTS["reduce_short"]

    def test_exit_short_count(self, backtest_outputs):
        count = (backtest_outputs.fills["fill_type"] == "exit_short").sum()

        assert count == self.EXPECTED_FILL_COUNTS["exit_short"]

    def test_profiles_enter_on_different_bars(self, backtest_outputs):
        fills = backtest_outputs.fills
        entries = fills[fills["fill_type"].str.contains("enter|add_to")]
        by_timestamp = entries.groupby(entries.index)["tag"].apply(set)
        staggered = [ts for ts, names in by_timestamp.items() if len(names) == 1]

        assert len(staggered) > 0

    def test_closed_trade_count(self, analyser):
        assert analyser.closed_trades_count == self.EXPECTED_CLOSED_TRADES

    def test_every_exited_quantity_is_a_closed_trade(self, backtest_outputs, analyser):
        fills = backtest_outputs.fills
        exited = fills.loc[
            fills["fill_type"].str.startswith(("exit", "reduce")), "net_quantity"
        ].sum()

        assert analyser.trades["net_quantity"].sum() == pytest.approx(exited)

    def test_final_equity(self, backtest_outputs):
        assert backtest_outputs.final_balance["USDT"] == pytest.approx(
            self.EXPECTED_FINAL_BALANCE, abs=self.TOLERANCE
        )

    def test_max_drawdown(self, analyser):
        assert analyser.max_drawdown == pytest.approx(
            self.EXPECTED_MAX_DRAWDOWN, abs=1e-6
        )
