import numpy as np
import pandas as pd

from robottraderslab.backtester import run_backtest
from robottraderslab.bootstrap import BotConfig

_CONFIG = """\
[backtest]
initial_balance = {{ USDT = 10000.0 }}
maker_fee_rate = 0.001
taker_fee_rate = 0.001
start_date = "2024-01-01"
end_date = "2024-04-01"
{fee_mode_line}

[backtest.ohlcv_provider]
ohlcv_provider = "mock"

[strategy]
strategy_class = "futures_ma"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1d"
available_balance_ratio = 0.5
fast_ma_length = 3
slow_ma_length = 7
"""

_NO_CONVENTION_NAMED = ""
_COST_NAMED = 'fee_mode = "cost"'


def _entry_fills(fee_mode_line: str) -> pd.DataFrame:
    """Both conventions must replay the same candles for the comparison to hold."""
    np.random.seed(42)
    config = BotConfig.from_text(_CONFIG.format(fee_mode_line=fee_mode_line))
    fills = run_backtest(config).fills
    return fills[fills["fill_type"].isin(["enter_long", "enter_short"])]


class TestTheConfiguredConventionReachesTheFills:
    """The mode a config names, or leaves to the default, is the mode every
    fill of the run is charged under.
    """

    def test_a_config_naming_no_convention_fills_the_quantity_ordered(self):
        entries = _entry_fills(_NO_CONVENTION_NAMED)

        assert len(entries) > 0
        assert (entries["net_quantity"] == entries["gross_quantity"]).all()

    def test_a_cost_config_shaves_the_fee_off_the_fill(self):
        entries = _entry_fills(_COST_NAMED)

        assert len(entries) > 0
        assert (entries["net_quantity"] < entries["gross_quantity"]).all()
