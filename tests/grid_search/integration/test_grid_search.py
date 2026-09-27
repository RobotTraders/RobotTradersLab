from pathlib import Path

import pandas as pd
import pytest

from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.grid_search import GridSearchResults, run_grid_search
from robottraderslab.grid_search.command import main as grid_search_main
from robottraderslab.grid_search.optimiser import GridSearchOptimiser

CSV_PATH = "../../ma_strategy/integration/single_symbol_pipeline/data/btc_usdt_binance_2024.csv"

TEST_CONFIG_TOML = f"""
[strategy]
strategy_class = "futures_ma"

[backtest]
initial_balance = {{ USDT = 10000.0 }}
maker_fee_rate = 0.001
taker_fee_rate = 0.001
fee_mode = "cost"
start_date = "2024-01-01"
end_date = "2024-12-31"

[backtest.ohlcv_provider]
ohlcv_provider = "csv"
file = "{CSV_PATH}"
symbol = "BTC/USDT:USDT"
timeframe = "1d"

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1d"
fast_ma_length = 5
slow_ma_length = 20
available_balance_ratio = 1.0

[optimisation]
max_workers = 1

[[optimisation.parameter_configs]]
parameter = "strategy.profiles.fast_ma_length"
sampling = "linear"
min = 5.0
max = 15.0
num_samples = 3

[[optimisation.parameter_configs]]
parameter = "strategy.profiles.slow_ma_length"
sampling = "linear"
min = 20.0
max = 40.0
num_samples = 2
"""

EXPECTED_GRID_SIZE = 6


class TestGridSearchIntegration:
    @pytest.fixture(scope="class")
    def results(self) -> GridSearchResults:
        bot_config = BotConfig.from_text(TEST_CONFIG_TOML)
        assert bot_config.backtest is not None
        csv_file = Path(__file__).parent / bot_config.backtest.ohlcv_provider["file"]
        bot_config.backtest.ohlcv_provider["file"] = str(csv_file)
        return run_grid_search(bot_config)

    def test_all_grid_points_succeeded(self, results):
        df = results.dataframe

        assert len(df) == EXPECTED_GRID_SIZE
        assert df["sharpe_ratio"].notna().all()

    def test_best_sharpe_ratio_at_fast_10_slow_20(self, results):
        df = results.dataframe

        best = df.loc[df["sharpe_ratio"].idxmax()]

        assert int(best["strategy.profiles.fast_ma_length"]) == 10
        assert int(best["strategy.profiles.slow_ma_length"]) == 20
        assert best["sharpe_ratio"] == pytest.approx(1.865038, abs=0.01)

    def test_best_return_at_fast_10_slow_20(self, results):
        df = results.dataframe

        best = df.loc[df["roi"].idxmax()]

        assert int(best["strategy.profiles.fast_ma_length"]) == 10
        assert int(best["strategy.profiles.slow_ma_length"]) == 20
        assert best["roi"] == pytest.approx(0.9507, abs=0.01)

    def test_known_metric_values(self, results):
        df = results.dataframe

        row = df[
            (df["strategy.profiles.fast_ma_length"] == 5)
            & (df["strategy.profiles.slow_ma_length"] == 40)
        ].iloc[0]

        assert row["sharpe_ratio"] == pytest.approx(-0.2474, abs=0.01)
        assert row["roi"] == pytest.approx(-0.1039, abs=0.01)
        assert row["max_drawdown"] == pytest.approx(-0.2329, abs=0.01)
        assert row["win_rate"] == pytest.approx(0.5, abs=0.01)

    def test_parameter_combinations_are_unique(self, results):
        df = results.dataframe

        param_pairs = df[
            ["strategy.profiles.fast_ma_length", "strategy.profiles.slow_ma_length"]
        ].drop_duplicates()

        assert len(param_pairs) == EXPECTED_GRID_SIZE

    def test_to_csv(self, results, tmp_path):
        csv_path = tmp_path / "results.csv"

        results.to_csv(csv_path)

        exported = pd.read_csv(csv_path)
        assert len(exported) == EXPECTED_GRID_SIZE

    def test_from_csv_roundtrip(self, results, tmp_path):
        csv_path = tmp_path / "results.csv"
        results.to_csv(csv_path)

        loaded = GridSearchResults.from_csv(csv_path)

        assert len(loaded.dataframe) == EXPECTED_GRID_SIZE
        assert "sharpe_ratio" in loaded.dataframe.columns
        assert "strategy.profiles.fast_ma_length" in loaded.dataframe.columns

        best = loaded.dataframe.loc[loaded.dataframe["sharpe_ratio"].idxmax()]
        assert int(best["strategy.profiles.fast_ma_length"]) == 10


class TestSweepOutcome:
    @pytest.fixture
    def config_every_point_scores(self, tmp_path: Path) -> Path:
        return _written(tmp_path, _with_resolved_file(TEST_CONFIG_TOML))

    @pytest.fixture
    def config_no_point_can_score(self, tmp_path: Path) -> Path:
        return _written(
            tmp_path,
            _on_a_symbol_the_provider_refuses(_with_resolved_file(TEST_CONFIG_TOML)),
        )

    def test_a_grid_with_scored_points_runs_to_the_end(self, config_every_point_scores):
        grid_search_main(config_every_point_scores, None)

    def test_the_run_stops_when_no_point_scored(self, config_no_point_can_score):
        with pytest.raises(
            StrategyCriticalError,
            match=f"No point of the {EXPECTED_GRID_SIZE}-point grid",
        ):
            grid_search_main(config_no_point_can_score, None)

    def test_the_results_file_is_written_before_the_run_stops(
        self, config_no_point_can_score, tmp_path
    ):
        exported = tmp_path / "results.csv"

        with pytest.raises(StrategyCriticalError):
            grid_search_main(config_no_point_can_score, exported)

        assert pd.read_csv(exported)["error"].notna().all()


def test_a_sweep_returns_the_same_results_at_one_worker_and_at_two():
    one_worker = BotConfig.from_text(_with_resolved_file(TEST_CONFIG_TOML))
    two_workers = BotConfig.from_text(
        _with_resolved_file(TEST_CONFIG_TOML).replace(
            "max_workers = 1", "max_workers = 2"
        )
    )

    sequential = GridSearchOptimiser(one_worker).run()
    pooled = GridSearchOptimiser(two_workers).run()

    assert sequential == pooled


def _with_resolved_file(toml: str) -> str:
    return toml.replace(
        f'file = "{CSV_PATH}"',
        f'file = "{(Path(__file__).parent / CSV_PATH).resolve().as_posix()}"',
    )


def _on_a_symbol_the_provider_refuses(toml: str) -> str:
    header = "[[strategy.profiles]]"
    before, profiles = toml.split(header, 1)
    return before + header + profiles.replace('"BTC/USDT:USDT"', '"ETH/USDT:USDT"', 1)


def _written(config_dir: Path, toml: str) -> Path:
    config = config_dir / "grid.toml"
    config.write_text(toml, encoding="utf-8")
    return config
