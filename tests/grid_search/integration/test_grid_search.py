import tomllib
from pathlib import Path

import pandas as pd
import pytest

from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import BotCriticalError
from robottraderslab.grid_search import GridSearchResults, run_grid_search
from robottraderslab.grid_search.command import main as grid_search_main
from robottraderslab.grid_search.optimisation_config import OptimisationConfig
from robottraderslab.grid_search.optimiser import GridSearchOptimiser

CSV_PATH = "../../ma_strategy/integration/single_symbol_pipeline/data/btc_usdt_binance_2024.csv"

NAME = 'name = "test-bot"\n'
BOT_TOML = f"""
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
"""

SEARCH_TOML = """
max_workers = 1

[profile]
symbol = "BTC/USDT:USDT"
timeframe = "1d"

[[parameter_configs]]
parameter = "fast_ma_length"
sampling = "linear"
min = 5.0
max = 15.0
num_samples = 3

[[parameter_configs]]
parameter = "slow_ma_length"
sampling = "linear"
min = 20.0
max = 40.0
num_samples = 2
"""

EXPECTED_GRID_SIZE = 6


class TestGridSearchIntegration:
    @pytest.fixture(scope="class")
    def results(self, tmp_path_factory) -> GridSearchResults:
        search_file = tmp_path_factory.mktemp("search") / "grid-search.toml"
        search_file.write_text(SEARCH_TOML, encoding="utf-8")
        bot_config = BotConfig.from_text(NAME + BOT_TOML)
        assert bot_config.backtest is not None
        csv_file = Path(__file__).parent / bot_config.backtest.ohlcv_provider["file"]
        bot_config.backtest.ohlcv_provider["file"] = str(csv_file)
        return run_grid_search(bot_config, search_file)

    def test_all_grid_points_succeeded(self, results):
        df = results.dataframe

        assert len(df) == EXPECTED_GRID_SIZE
        assert df["sharpe_ratio"].notna().all()

    def test_best_sharpe_ratio_at_fast_10_slow_20(self, results):
        df = results.dataframe

        best = df.loc[df["sharpe_ratio"].idxmax()]

        assert int(best["fast_ma_length"]) == 10
        assert int(best["slow_ma_length"]) == 20
        assert best["sharpe_ratio"] == pytest.approx(1.865038, abs=0.01)

    def test_best_return_at_fast_10_slow_20(self, results):
        df = results.dataframe

        best = df.loc[df["roi"].idxmax()]

        assert int(best["fast_ma_length"]) == 10
        assert int(best["slow_ma_length"]) == 20
        assert best["roi"] == pytest.approx(0.9507, abs=0.01)

    def test_known_metric_values(self, results):
        df = results.dataframe

        row = df[(df["fast_ma_length"] == 5) & (df["slow_ma_length"] == 40)].iloc[0]

        assert row["sharpe_ratio"] == pytest.approx(-0.2474, abs=0.01)
        assert row["roi"] == pytest.approx(-0.1039, abs=0.01)
        assert row["max_drawdown"] == pytest.approx(-0.2329, abs=0.01)
        assert row["win_rate"] == pytest.approx(0.5, abs=0.01)

    def test_parameter_combinations_are_unique(self, results):
        df = results.dataframe

        param_pairs = df[["fast_ma_length", "slow_ma_length"]].drop_duplicates()

        assert len(param_pairs) == EXPECTED_GRID_SIZE

    def test_to_csv(self, results, tmp_path):
        csv_path = tmp_path / "results.csv"

        results.to_csv(csv_path)

        exported = pd.read_csv(csv_path)
        assert len(exported) == EXPECTED_GRID_SIZE
        assert exported["sortino_ratio"].notna().all()
        assert exported["calmar_ratio"].notna().all()

    def test_from_csv_roundtrip(self, results, tmp_path):
        csv_path = tmp_path / "results.csv"
        results.to_csv(csv_path)

        loaded = GridSearchResults.from_csv(csv_path)

        assert len(loaded.dataframe) == EXPECTED_GRID_SIZE
        assert "sharpe_ratio" in loaded.dataframe.columns
        assert "fast_ma_length" in loaded.dataframe.columns

        best = loaded.dataframe.loc[loaded.dataframe["sharpe_ratio"].idxmax()]
        assert int(best["fast_ma_length"]) == 10


class TestSweepOutcome:
    @pytest.fixture
    def config_every_point_scores(self, tmp_path: Path) -> Path:
        return _written(tmp_path, _with_resolved_file(BOT_TOML), SEARCH_TOML)

    @pytest.fixture
    def config_no_point_can_score(self, tmp_path: Path) -> Path:
        return _written(
            tmp_path,
            _on_a_symbol_the_provider_refuses(_with_resolved_file(BOT_TOML)),
            SEARCH_TOML.replace("BTC/USDT:USDT", "ETH/USDT:USDT"),
        )

    @pytest.fixture
    def config_no_point_trades(self, tmp_path: Path) -> Path:
        return _written(
            tmp_path,
            _with_resolved_file(BOT_TOML),
            SEARCH_TOML.replace("min = 20.0\nmax = 40.0", "min = 364.0\nmax = 366.0"),
        )

    def test_a_grid_where_no_point_trades_scores_flat_and_runs_to_the_end(
        self, config_no_point_trades, tmp_path, capsys
    ):
        exported = tmp_path / "results.csv"

        grid_search_main(config_no_point_trades, exported)

        flat_rows = pd.read_csv(exported)
        assert flat_rows["closed_trades"].tolist() == [0] * EXPECTED_GRID_SIZE
        assert flat_rows["error"].isna().all()
        console = capsys.readouterr().out
        for position in range(1, EXPECTED_GRID_SIZE + 1):
            assert (
                f"INFO: Grid point {position}/{EXPECTED_GRID_SIZE} completed" in console
            )
        assert "ERROR" not in console
        assert "Grid search completed: 6/6 successful" in console

    def test_a_sweep_keeps_the_run_in_a_folder_of_the_bot(
        self, config_every_point_scores
    ):
        grid_search_main(config_every_point_scores, None)

        (sweep,) = (config_every_point_scores / "grid-searches").iterdir()
        assert {path.name for path in sweep.iterdir()} == {
            "grid-search.toml",
            "bot.toml",
            "grid-search.log",
            "results.csv",
        }
        for name in ("grid-search.toml", "bot.toml"):
            assert (sweep / name).read_bytes() == (
                config_every_point_scores / name
            ).read_bytes()
        assert len(pd.read_csv(sweep / "results.csv")) == EXPECTED_GRID_SIZE
        assert "Grid search completed" in (sweep / "grid-search.log").read_text(
            encoding="utf-8"
        )

    def test_the_run_stops_when_no_point_scored(self, config_no_point_can_score):
        with pytest.raises(
            BotCriticalError,
            match=f"No point of the {EXPECTED_GRID_SIZE}-point grid",
        ):
            grid_search_main(config_no_point_can_score, None)

    def test_the_results_file_is_written_before_the_run_stops(
        self, config_no_point_can_score, tmp_path
    ):
        exported = tmp_path / "results.csv"

        with pytest.raises(BotCriticalError):
            grid_search_main(config_no_point_can_score, exported)

        assert pd.read_csv(exported)["error"].notna().all()


def test_a_sweep_returns_the_same_results_at_one_worker_and_at_two():
    bot = BotConfig.from_text(NAME + _with_resolved_file(BOT_TOML))
    one_worker = OptimisationConfig.from_config(tomllib.loads(SEARCH_TOML))
    two_workers = OptimisationConfig.from_config(
        tomllib.loads(SEARCH_TOML.replace("max_workers = 1", "max_workers = 2"))
    )

    sequential = GridSearchOptimiser(bot, one_worker).run()
    pooled = GridSearchOptimiser(bot, two_workers).run()

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


def _written(bot_folder: Path, bot_toml: str, search_toml: str) -> Path:
    (bot_folder / "bot.toml").write_text(bot_toml, encoding="utf-8")
    (bot_folder / "grid-search.toml").write_text(search_toml, encoding="utf-8")
    return bot_folder


@pytest.mark.parametrize("section", ["strategy", "backtest", "live"])
def test_a_search_carrying_a_section_of_the_bot_is_refused_naming_bot_toml(
    tmp_path, section
):
    folder = _written(
        tmp_path,
        _with_resolved_file(BOT_TOML),
        f"[{section}]\n" + SEARCH_TOML,
    )

    with pytest.raises(BotCriticalError, match="bot.toml") as refusal:
        grid_search_main(folder, None)

    assert f"[{section}]" in str(refusal.value)
    assert not (folder / "grid-searches").exists()
