from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import robottraderslab.live.report.command
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.live.report.command import main


@pytest.fixture
def bot_config() -> BotConfig:
    return BotConfig.model_validate(
        {"strategy": {"strategy_class": "fake-strategy-99"}}
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


def test_the_config_comes_from_the_given_file(read_bot_config):
    main(config=Path("my-filename.toml"), days=90)

    read_bot_config.assert_called_once_with(Path("my-filename.toml"))


def test_the_report_covers_the_requested_window(bot_config, run_report):
    main(config=Path("settings.toml"), days=7)

    run_report.assert_called_once_with(bot_config, days=7)


def test_a_config_file_that_cannot_be_read(read_bot_config, run_report):
    read_bot_config.side_effect = StrategyCriticalError("There is no config file")

    with pytest.raises(StrategyCriticalError, match="There is no config file"):
        main(config=Path("missing.toml"), days=90)

    run_report.assert_not_called()
