import pytest

from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.grid_search import GridSearchResults, run_grid_search

TOML_CONFIG = """\
[backtest]
initial_balance = { USDT = 10000.0 }
maker_fee_rate = 0.001
taker_fee_rate = 0.001
start_date = "2024-03-01"
end_date = "2024-04-01"
verbose = false

[backtest.ohlcv_provider]
ohlcv_provider = "mock"

[strategy]
strategy_class = "futures_ma"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "4h"
available_balance_ratio = 0.5
fast_ma_length = 3
slow_ma_length = 7
long_only = true

[optimisation]
max_workers = 2

[[optimisation.parameter_configs]]
parameter = "strategy.profiles.fast_ma_length"
sampling = "linear"
min = 3
max = 5
num_samples = 2

[[optimisation.parameter_configs]]
parameter = "strategy.profiles.slow_ma_length"
sampling = "linear"
min = 7
max = 10
num_samples = 2
"""

EXPECTED_GRID_SIZE = 4


class TestParallelGridSearch:
    @pytest.fixture(scope="class")
    def bot_config(self) -> BotConfig:
        return BotConfig.from_text(TOML_CONFIG)

    @pytest.fixture(scope="class")
    def results(self, bot_config: BotConfig) -> GridSearchResults:
        return run_grid_search(bot_config)

    def test_all_grid_points_return_results(self, results):
        assert len(results.dataframe) == EXPECTED_GRID_SIZE

    def test_all_results_have_metrics(self, results):
        df = results.dataframe

        assert len(df) == EXPECTED_GRID_SIZE

        assert df["roi"].notna().all()
        assert df["max_drawdown"].notna().all()

    def test_parameter_columns_present(self, results):
        df = results.dataframe

        assert "strategy.profiles.fast_ma_length" in df.columns
        assert "strategy.profiles.slow_ma_length" in df.columns

    def test_results_follow_the_grid_order(self, results):
        df = results.dataframe

        measured_points = list(
            zip(
                df["strategy.profiles.fast_ma_length"],
                df["strategy.profiles.slow_ma_length"],
                strict=True,
            )
        )

        assert measured_points == [(3, 7), (3, 10), (5, 7), (5, 10)]


def test_an_undeclared_setting_stops_the_sweep_before_any_point_is_measured(caplog):
    bot_config = BotConfig.from_text(
        TOML_CONFIG.replace("max_workers = 2", "max_workers = 1").replace(
            'parameter = "strategy.profiles.fast_ma_length"',
            'parameter = "strategy.profiles.no_such_setting"',
        )
    )

    with pytest.raises(StrategyCriticalError, match="no_such_setting"):
        run_grid_search(bot_config)

    assert not [record for record in caplog.records if "Grid point" in record.message]


def test_a_critical_stops_the_parallel_sweep():
    bot_config = BotConfig.from_text(
        TOML_CONFIG.replace(
            'parameter = "strategy.profiles.fast_ma_length"',
            'parameter = "strategy.profiles.no_such_setting"',
        )
    )

    with pytest.raises(StrategyCriticalError, match="no_such_setting"):
        run_grid_search(bot_config)
