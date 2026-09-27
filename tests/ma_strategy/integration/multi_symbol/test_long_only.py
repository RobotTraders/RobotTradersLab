import pytest

from robottraderslab.bootstrap import BotConfig

EXPECTED_SYMBOLS = {
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
}
SYMBOLS_1D = {"BTC/USDT:USDT", "ETH/USDT:USDT"}
SYMBOLS_4H = {"SOL/USDT:USDT", "XRP/USDT:USDT"}

TOML_CONFIG = """\
[backtest]
initial_balance = { USDT = 10000.0 }
maker_fee_rate = 0.001
taker_fee_rate = 0.002
fee_mode = "cost"
start_date = "2024-03-01"
end_date = "2024-06-01"
verbose = false

[backtest.ohlcv_provider]
ohlcv_provider = "mock"

[strategy]
strategy_class = "futures_ma"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1d"
available_balance_ratio = 0.25
fast_ma_length = 7
slow_ma_length = 20
long_only = true

[[strategy.profiles]]
symbol = "ETH/USDT:USDT"
timeframe = "1d"
available_balance_ratio = 0.25
fast_ma_length = 5
slow_ma_length = 15
long_only = true

[[strategy.profiles]]
symbol = "SOL/USDT:USDT"
timeframe = "4h"
available_balance_ratio = 0.25
fast_ma_length = 3
slow_ma_length = 7
long_only = true

[[strategy.profiles]]
symbol = "XRP/USDT:USDT"
timeframe = "4h"
available_balance_ratio = 0.25
fast_ma_length = 4
slow_ma_length = 10
long_only = true
"""


@pytest.fixture(scope="class")
def bot_config() -> BotConfig:
    return BotConfig.from_text(TOML_CONFIG)


class TestMultiSymbolLongOnly:
    """Verify multi-symbol long-only mode with MA crossover signals."""

    EXPECTED_TRADE_COUNT = 69
    EXPECTED_ROI = -0.0047
    EXPECTED_MAX_DRAWDOWN = -0.0408
    EXPECTED_WIN_RATE = 0.3478
    EXPECTED_PROFIT_FACTOR = 0.9080
    EXPECTED_TOTAL_PNL = -127.0886
    TOLERANCE = 0.001

    def test_all_symbols_traded(self, backtest_outputs):
        traded_symbols = set(backtest_outputs.fills["symbol"].unique())

        assert traded_symbols == EXPECTED_SYMBOLS

    def test_only_long_fill_types(self, backtest_outputs):
        fill_types = set(backtest_outputs.fills["fill_type"].unique())

        assert fill_types == {"enter_long", "exit_long"}

    def test_fill_reasons(self, backtest_outputs):
        reasons = set(backtest_outputs.fills["reason"].unique())

        assert reasons == {"ma crossover buy", "ma crossover sell"}

    def test_4h_symbols_have_more_fills_than_1d(self, backtest_outputs):
        fills = backtest_outputs.fills
        fills_4h = len(fills[fills["symbol"].isin(SYMBOLS_4H)])
        fills_1d = len(fills[fills["symbol"].isin(SYMBOLS_1D)])

        assert fills_4h > fills_1d

    def test_trade_count(self, analyser):
        assert analyser.closed_trades_count == self.EXPECTED_TRADE_COUNT

    def test_roi(self, analyser):
        assert abs(analyser.roi - self.EXPECTED_ROI) < self.TOLERANCE

    def test_max_drawdown(self, analyser):
        assert abs(analyser.max_drawdown - self.EXPECTED_MAX_DRAWDOWN) < self.TOLERANCE

    def test_win_rate(self, analyser):
        assert abs(analyser.win_rate - self.EXPECTED_WIN_RATE) < self.TOLERANCE

    def test_profit_factor(self, analyser):
        assert (
            abs(analyser.profit_factor - self.EXPECTED_PROFIT_FACTOR) < self.TOLERANCE
        )

    def test_total_pnl(self, analyser):
        assert abs(analyser.total_pnl - self.EXPECTED_TOTAL_PNL) < self.TOLERANCE
