from pathlib import Path

from robottraderslab.bootstrap import BotConfig, resolve_path


def test_resolves_relative_value_against_config_dir(tmp_path):
    toml_file = tmp_path / "workspace" / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir(parents=True)
    toml_file.write_text(
        '[strategy]\nstrategy_class = "dummy"\nmodel_dir = "results/run"\n'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    resolved = resolve_path(bot_config, "strategy.model_dir")
    assert resolved == toml_file.parent / "results" / "run"


def test_returns_absolute_value_unchanged(tmp_path):
    absolute_path = (tmp_path / "models" / "btc.pkl").as_posix()
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        f'[strategy]\nstrategy_class = "dummy"\nmodel_file = "{absolute_path}"\n'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    resolved = resolve_path(bot_config, "strategy.model_file")
    assert resolved == Path(absolute_path)


def test_returns_none_for_missing_key(tmp_path):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text('[strategy]\nstrategy_class = "dummy"\nname = "demo"\n')

    bot_config = BotConfig.from_file(str(toml_file))

    assert resolve_path(bot_config, "strategy.model_dir") is None
