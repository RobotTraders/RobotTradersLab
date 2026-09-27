from pathlib import Path

import pytest

from robottraderslab._core import to_quantity
from robottraderslab.bootstrap import BotConfig

TOML_CONFIG = """\
[backtest]
initial_balance = { USDT = 10000.0 }
maker_fee_rate = 0.001
taker_fee_rate = 0.001
fee_mode = "exchange"
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
fast_ma_length = 2
slow_ma_length = 8
trend_ma_length = 40
tag = "fast"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1d"
available_balance_ratio = 0.4
fast_ma_length = 5
slow_ma_length = 20
trend_ma_length = 40
tag = "slow"
"""

ONE_QUANTITY_STEP = to_quantity(1)


@pytest.fixture(scope="class")
def bot_config() -> BotConfig:
    bot_config = BotConfig.from_text(TOML_CONFIG)
    assert bot_config.backtest is not None
    csv_file = Path(__file__).parent / bot_config.backtest.ohlcv_provider["file"]
    bot_config.backtest.ohlcv_provider["file"] = str(csv_file)
    return bot_config


class TestStackedProfilesTradeTable:
    """Two profiles flip sides through the year, so an exit of one covers an
    entry of the other only in part.
    """

    def test_no_reported_trade_holds_less_than_one_quantity_step(
        self, trade_aggregator
    ):
        assert trade_aggregator.trades["net_quantity"].min() >= ONE_QUANTITY_STEP
