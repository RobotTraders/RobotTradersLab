from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import robottraderslab.live.report.command
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import BotCriticalError
from robottraderslab.live.report.command import main


@pytest.fixture
def bot_config() -> BotConfig:
    return BotConfig.model_validate(
        {"name": "test-bot", "strategy": {"strategy_class": "fake-strategy-99"}}
    )


@pytest.fixture(autouse=True)
def read_bot_config(bot_config: BotConfig) -> Iterator[Mock]:
    with patch.object(
        robottraderslab.live.report.command.BotConfig, "from_file", autospec=True
    ) as loaded:
        loaded.return_value = bot_config
        yield loaded


@pytest.fixture(autouse=True)
def run_report() -> Iterator[Mock]:
    with patch(
        "robottraderslab.live.report.command.run_report", autospec=True
    ) as charted:
        yield charted


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    (tmp_path / "bot.toml").write_text("")
    return tmp_path


def test_the_config_comes_from_the_bot_toml_of_the_given_folder(
    read_bot_config, folder
):
    main(folder=folder, days=90, save=False)

    read_bot_config.assert_called_once_with(folder / "bot.toml")


def test_the_report_covers_the_requested_window(bot_config, run_report, folder):
    main(folder=folder, days=7, save=False)

    run_report.assert_called_once_with(bot_config, days=7, save=False)


def test_a_config_file_that_cannot_be_read(read_bot_config, run_report, folder):
    read_bot_config.side_effect = BotCriticalError("There is no config file")

    with pytest.raises(BotCriticalError, match="There is no config file"):
        main(folder=folder, days=90, save=False)

    run_report.assert_not_called()
