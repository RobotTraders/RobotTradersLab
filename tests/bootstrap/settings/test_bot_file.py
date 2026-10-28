import pytest

from robottraderslab.bootstrap import bot_file, optimisation_file
from robottraderslab.exceptions import BotCriticalError


def test_a_folder_holding_bot_toml_gives_its_file(tmp_path):
    (tmp_path / "bot.toml").write_text("", encoding="utf-8")

    assert bot_file(tmp_path) == tmp_path / "bot.toml"


def test_a_folder_without_bot_toml_names_the_file_and_the_folder(tmp_path):
    with pytest.raises(BotCriticalError, match="bot.toml") as refusal:
        bot_file(tmp_path)

    assert str(tmp_path) in str(refusal.value)


def test_a_toml_file_is_refused_naming_its_folder(tmp_path):
    config = tmp_path / "trend" / "bot_b.toml"
    config.parent.mkdir()
    config.write_text("", encoding="utf-8")

    with pytest.raises(BotCriticalError, match="a bot is a folder") as refusal:
        bot_file(config)

    assert f"`{config.parent}`" in str(refusal.value)


def test_a_toml_path_whose_file_is_gone_is_refused_naming_its_folder(tmp_path):
    config = tmp_path / "trend" / "bot_b.toml"

    with pytest.raises(BotCriticalError, match="a bot is a folder") as refusal:
        bot_file(config)

    assert f"`{config.parent}`" in str(refusal.value)


def test_a_folder_holding_optimisation_toml_gives_its_file(tmp_path):
    (tmp_path / "optimisation.toml").write_text("", encoding="utf-8")

    assert optimisation_file(tmp_path) == tmp_path / "optimisation.toml"


def test_a_folder_without_optimisation_toml_names_the_file(tmp_path):
    (tmp_path / "bot.toml").write_text("", encoding="utf-8")

    with pytest.raises(BotCriticalError, match="optimisation.toml"):
        optimisation_file(tmp_path)
