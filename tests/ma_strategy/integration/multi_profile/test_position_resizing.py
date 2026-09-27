from pathlib import Path

import pytest

from robottraderslab.backtester import BacktestOutputs
from robottraderslab.bootstrap import BotConfig

TOML_CONFIG = """\
[backtest]
initial_balance = { USDT = 10000.0 }
maker_fee_rate = 0.001
taker_fee_rate = 0.001
fee_mode = "cost"
start_date = "2024-01-01"
end_date = "2024-01-30"
verbose = false

[backtest.ohlcv_provider]
ohlcv_provider = "csv"
file = "data/resizing_test.csv"
symbol = "BTC/USDT:USDT"
timeframe = "1d"

[strategy]
strategy_class = "futures_ma"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1d"
available_balance_ratio = 0.25
fast_ma_length = 2
slow_ma_length = 3
trend_ma_length = 6
tag = "alpha"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1d"
available_balance_ratio = 0.25
fast_ma_length = 3
slow_ma_length = 4
trend_ma_length = 6
tag = "beta"
"""

INITIAL_BALANCE = 10000.0
AVAILABLE_BALANCE_RATIO = 0.25
FEE_RATE = 0.001
TOLERANCE = 0.01


@pytest.fixture(scope="class")
def bot_config() -> BotConfig:
    bot_config = BotConfig.from_text(TOML_CONFIG)
    assert bot_config.backtest is not None
    csv_file = Path(__file__).parent / bot_config.backtest.ohlcv_provider["file"]
    bot_config.backtest.ohlcv_provider["file"] = str(csv_file)
    return bot_config


class TestPositionResizing:
    """Verify intermediate position state and balance-ratio sizing across stacked fills.

    Uses crafted CSV data (data/resizing_test.csv) with two profiles on BTC/USDT:USDT@1d:
    - alpha (fast=2, slow=3): reacts first
    - beta (fast=3, slow=4): reacts second, producing add_to/reduce fills

    Expected fill sequence:
    1. enter_long (alpha)  → add_to_long (beta)  → reduce_long (alpha) → exit_long (beta)
    2. enter_short (alpha) → add_to_short (beta) → reduce_short (alpha) → exit_short (beta)
    """

    EXPECTED_FILL_SEQUENCE = [
        ("enter_long", "alpha"),
        ("add_to_long", "beta"),
        ("reduce_long", "alpha"),
        ("exit_long", "beta"),
        ("enter_short", "alpha"),
        ("add_to_short", "beta"),
        ("reduce_short", "alpha"),
        ("exit_short", "beta"),
    ]
    EXPECTED_FINAL_BALANCE = 13029.588740712361

    def test_fill_sequence(self, backtest_outputs):
        fills = backtest_outputs.fills
        actual = [
            (row["fill_type"], row["tag"].split("-")[-1]) for _, row in fills.iterrows()
        ]

        assert actual == self.EXPECTED_FILL_SEQUENCE

    def test_enter_long_quantity_matches_initial_available_balance_ratio(
        self, backtest_outputs
    ):
        enter_long = backtest_outputs.fills.iloc[0]
        expected_qty = INITIAL_BALANCE * AVAILABLE_BALANCE_RATIO / enter_long["price"]

        assert enter_long["gross_quantity"] == pytest.approx(expected_qty, rel=1e-6)

    def test_add_to_long_quantity_less_than_enter_long(self, backtest_outputs):
        enter_long = backtest_outputs.fills.iloc[0]
        add_to_long = backtest_outputs.fills.iloc[1]

        assert add_to_long["gross_quantity"] < enter_long["gross_quantity"]

    def test_add_to_long_quantity_matches_reduced_available_balance(
        self,
        backtest_outputs: BacktestOutputs,
    ):
        add_to_long = backtest_outputs.fills.iloc[1]
        available_after_enter = (
            backtest_outputs.trade_equity_snapshots[0].balances["USDT"].available
        )
        expected_qty = (
            available_after_enter * AVAILABLE_BALANCE_RATIO / add_to_long["price"]
        )

        assert add_to_long["gross_quantity"] == pytest.approx(expected_qty, rel=1e-6)

    def test_balance_drops_after_enter_long(self, backtest_outputs: BacktestOutputs):
        after_enter = backtest_outputs.trade_equity_snapshots[0]
        available = after_enter.balances["USDT"].available

        assert available == pytest.approx(7500.0, abs=TOLERANCE)

    def test_balance_recovers_after_reduce_long(
        self,
        backtest_outputs: BacktestOutputs,
    ):
        after_enter = backtest_outputs.trade_equity_snapshots[0]
        after_reduce = backtest_outputs.trade_equity_snapshots[2]

        assert (
            after_reduce.balances["USDT"].available
            > after_enter.balances["USDT"].available
        )

    def test_balance_fully_unlocked_after_exit_long(
        self,
        backtest_outputs: BacktestOutputs,
    ):
        after_exit = backtest_outputs.trade_equity_snapshots[3]

        assert after_exit.balances["USDT"].locked == pytest.approx(0.0, abs=TOLERANCE)

    def test_enter_short_quantity_reflects_post_long_balance(
        self,
        backtest_outputs: BacktestOutputs,
    ):
        enter_short = backtest_outputs.fills.iloc[4]
        available_before_short = (
            backtest_outputs.trade_equity_snapshots[3].balances["USDT"].available
        )
        expected_qty = (
            available_before_short * AVAILABLE_BALANCE_RATIO / enter_short["price"]
        )

        assert enter_short["gross_quantity"] == pytest.approx(expected_qty, rel=1e-6)

    def test_add_to_short_quantity_less_than_enter_short(self, backtest_outputs):
        enter_short = backtest_outputs.fills.iloc[4]
        add_to_short = backtest_outputs.fills.iloc[5]

        assert add_to_short["gross_quantity"] < enter_short["gross_quantity"]

    def test_balance_fully_unlocked_after_exit_short(
        self,
        backtest_outputs: BacktestOutputs,
    ):
        after_exit_short = backtest_outputs.trade_equity_snapshots[7]

        assert after_exit_short.balances["USDT"].locked == pytest.approx(
            0.0, abs=TOLERANCE
        )

    def test_reduce_long_exits_alpha_tracked_quantity(self, backtest_outputs):
        enter_long = backtest_outputs.fills.iloc[0]
        reduce_long = backtest_outputs.fills.iloc[2]
        expected_qty = enter_long["gross_quantity"] * (1 - FEE_RATE)

        assert reduce_long["gross_quantity"] == pytest.approx(expected_qty, rel=1e-6)

    def test_exit_long_exits_beta_tracked_quantity(self, backtest_outputs):
        add_to_long = backtest_outputs.fills.iloc[1]
        exit_long = backtest_outputs.fills.iloc[3]
        expected_qty = add_to_long["gross_quantity"] * (1 - FEE_RATE)

        assert exit_long["gross_quantity"] == pytest.approx(expected_qty, rel=1e-6)

    def test_equity_includes_unrealised_pnl_after_enter_long(
        self,
        backtest_outputs: BacktestOutputs,
    ):
        after_enter = backtest_outputs.trade_equity_snapshots[0]
        equity = after_enter.get_equity("USDT")
        balance_total = after_enter.balances["USDT"].total

        assert equity == pytest.approx(9997.5, abs=TOLERANCE)
        assert equity == pytest.approx(balance_total, abs=TOLERANCE)

    def test_equity_diverges_from_balance_after_add_to_long(
        self,
        backtest_outputs: BacktestOutputs,
    ):
        after_add = backtest_outputs.trade_equity_snapshots[1]
        equity = after_add.get_equity("USDT")
        balance_total = after_add.balances["USDT"].total

        assert equity == pytest.approx(10235.769230769231, abs=TOLERANCE)
        assert equity > balance_total

    def test_equity_equals_balance_when_flat_after_long_cycle(
        self,
        backtest_outputs: BacktestOutputs,
    ):
        after_exit_long = backtest_outputs.trade_equity_snapshots[3]
        equity = after_exit_long.get_equity("USDT")

        assert equity == pytest.approx(after_exit_long.balances["USDT"].total, abs=1e-6)
        assert len(after_exit_long.positions) == 0

    def test_equity_equals_balance_when_flat_after_short_cycle(
        self,
        backtest_outputs: BacktestOutputs,
    ):
        after_exit_short = backtest_outputs.trade_equity_snapshots[7]
        equity = after_exit_short.get_equity("USDT")

        assert equity == pytest.approx(
            after_exit_short.balances["USDT"].total, abs=1e-6
        )
        assert len(after_exit_short.positions) == 0

    def test_long_exit_precedes_short_entry(self, backtest_outputs):
        fills = backtest_outputs.fills
        last_long_exit = fills[fills["fill_type"] == "exit_long"].index[-1]
        first_short_entry = fills[fills["fill_type"] == "enter_short"].index[0]

        assert last_long_exit < first_short_entry

    def test_no_position_between_long_and_short_cycles(
        self,
        backtest_outputs: BacktestOutputs,
    ):
        after_exit_long = backtest_outputs.trade_equity_snapshots[3]

        assert len(after_exit_long.positions) == 0

    def test_entry_net_quantity_accounts_for_fees(self, backtest_outputs):
        entries = backtest_outputs.fills[
            backtest_outputs.fills["fill_type"].str.startswith("enter")
        ]
        for _, fill in entries.iterrows():
            expected_net = fill["gross_quantity"] * (1 - FEE_RATE)

            assert fill["net_quantity"] == pytest.approx(expected_net, rel=1e-6)

    def test_final_balance(self, backtest_outputs):
        assert backtest_outputs.final_balance["USDT"] == pytest.approx(
            self.EXPECTED_FINAL_BALANCE, abs=TOLERANCE
        )
