from collections.abc import Callable
from importlib.resources import as_file, files
from pathlib import Path

import pytest

from robottraderslab import BotConfig
from robottraderslab.grid_search import main as grid_search
from robottraderslab.grid_search.optimisation_config import OptimisationConfig
from robottraderslab.scaffold import main as new


@pytest.fixture
def load_example() -> Callable[[str], BotConfig]:
    def load(name: str) -> BotConfig:
        config_path = files("robottraderslab_impulse") / "examples" / name
        with as_file(config_path) as path:
            return BotConfig.from_file(str(path))

    return load


def test_backtest_example_config_loads(load_example):
    bot_config = load_example("bot.toml")

    assert bot_config.strategy.strategy_class == "impulse"


def test_grid_search_example_declares_two_parameters_of_the_bot():
    with as_file(files("robottraderslab_impulse") / "examples") as examples:
        search = OptimisationConfig.from_file(examples / "grid-search.toml")

    assert [declared.parameter for declared in search.parameter_configs] == [
        "trix_length",
        "signal_length",
    ]
    assert search.profile.symbol == "ETH/USDT:USDT"
    assert (search.profile.timeframe, search.profile.tag) == ("4h", "alpha")


def test_grid_search_example_sweeps_whole_lengths_of_one_profile(load_example):
    with as_file(files("robottraderslab_impulse") / "examples") as examples:
        search = OptimisationConfig.from_file(examples / "grid-search.toml")
    profiles = load_example("bot.toml").strategy.profiles
    profile = profiles[search.profile.index_in(profiles)]

    assert isinstance(profile["trix_length"], int)
    assert isinstance(profile["signal_length"], int)


def test_new_creates_a_bot_folder_with_its_notebooks(tmp_path):
    new("impulse", Path("first-bot"), tmp_path)
    folder = tmp_path / "first-bot"

    assert {path.name for path in folder.iterdir()} >= {
        "bot.toml",
        "grid-search.toml",
        "run_backtest.ipynb",
        "run_grid_search.ipynb",
    }


def test_a_grid_search_on_the_copied_example_writes_its_run_folder(tmp_path):
    new("impulse", Path("first-bot"), tmp_path)
    folder = tmp_path / "first-bot"
    bot = folder / "bot.toml"
    bot.write_text(
        bot.read_text(encoding="utf-8")
        .replace('"ccxt_bitget"', '"mock"')
        .replace('start_date = "2025-04-01"', 'start_date = "2026-03-01"'),
        encoding="utf-8",
    )
    search = folder / "grid-search.toml"
    search.write_text(
        "max_workers = 1\n\n"
        + search.read_text(encoding="utf-8").replace(
            "num_samples = 10", "num_samples = 2"
        ),
        encoding="utf-8",
    )

    grid_search(folder, None)

    (run,) = (folder / "grid-searches").iterdir()
    assert {path.name for path in run.iterdir()} == {
        "grid-search.toml",
        "bot.toml",
        "grid-search.log",
        "results.csv",
    }
    for name in ("grid-search.toml", "bot.toml"):
        assert (run / name).read_bytes() == (folder / name).read_bytes()
