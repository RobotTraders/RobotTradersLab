import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

import robottraderslab.live.command
from robottraderslab._core import EX_TEMPFAIL, hold_instance_lock
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import ExchangeCriticalError, StrategyCriticalError
from robottraderslab.live.command import main

DECLARED = {
    "strategy": {"strategy_class": "fake-strategy-99"},
    "live": {
        "trading_account": {"account_name": "demoaccount"},
        "ohlcv_provider": {"name": "ccxt"},
    },
}


@pytest.fixture
def bot_config() -> BotConfig:
    return BotConfig.model_validate(DECLARED)


@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def setup_logging() -> Iterator[Mock]:
    with patch(
        "robottraderslab.live.command.setup_live_logging", autospec=True
    ) as configured:
        yield configured


@pytest.fixture(autouse=True)
def read_bot_config(bot_config: BotConfig) -> Iterator[Mock]:
    with patch.object(
        robottraderslab.live.command.BotConfig, "from_file", autospec=True
    ) as loaded:
        loaded.return_value = bot_config
        yield loaded


@pytest.fixture(autouse=True)
def run_livebot() -> Iterator[Mock]:
    with patch("robottraderslab.live.command.run_livebot", autospec=True) as ran:
        yield ran


def test_logging_falls_back_to_the_config_file(setup_logging):
    _run()

    setup_logging.assert_called_once_with(
        logfile=None,
        console_level=logging.getLevelName(logging.INFO),
        file_level=logging.getLevelName(logging.INFO),
        enable_console=False,
        enable_file=True,
        retention_days=60,
        custom_handlers={},
        config_dir=None,
        log_name="live_bot-example",
    )


def test_the_default_log_is_named_after_the_bot_file(setup_logging):
    _run(config=Path("impulse-bitget.toml"))

    assert setup_logging.call_args.kwargs["log_name"] == "live_impulse-bitget"


def test_the_given_logfile_wins_over_the_config_file(setup_logging):
    _run(logfile_override=Path("my-logfile.log"))

    setup_logging.assert_called_once()
    assert setup_logging.call_args.kwargs["logfile"] == Path("my-logfile.log")


def test_retention_comes_from_the_config_file(read_bot_config, setup_logging):
    declared = {
        **DECLARED,
        "live": {**DECLARED["live"], "logging": {"retention_days": 7}},
    }
    read_bot_config.return_value = BotConfig.model_validate(declared)

    _run()

    assert setup_logging.call_args.kwargs["retention_days"] == 7


def test_the_given_retention_wins_over_the_config_file(read_bot_config, setup_logging):
    declared = {
        **DECLARED,
        "live": {**DECLARED["live"], "logging": {"retention_days": 7}},
    }
    read_bot_config.return_value = BotConfig.model_validate(declared)

    _run(logfiles_retention_days=3)

    assert setup_logging.call_args.kwargs["retention_days"] == 3


def test_the_config_comes_from_the_given_file(read_bot_config):
    _run(config=Path("my-filename.toml"))

    read_bot_config.assert_called_once_with(Path("my-filename.toml"))


def test_run_livebot_is_called_with_the_loaded_config(bot_config, run_livebot):
    _run()

    run_livebot.assert_called_once_with(bot_config)


def test_a_config_file_that_cannot_be_read(read_bot_config, run_livebot):
    read_bot_config.side_effect = StrategyCriticalError("no config file at `nope`")

    with pytest.raises(StrategyCriticalError, match="no config file"):
        _run()

    run_livebot.assert_not_called()


def test_a_log_occasion_secret_is_resolved_before_logging_setup(
    read_bot_config, setup_logging, tmp_path
):
    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(
        '[[secrets]]\nname = "discord_main"\nwebhook_url = "https://from-secret"\n'
    )
    declared = {
        **DECLARED,
        "secrets_file": str(secrets_file),
        "live": {
            **DECLARED["live"],
            "notifier": {"discord": {"log": {"secret_name": "discord_main"}}},
        },
    }
    read_bot_config.return_value = BotConfig.model_validate(declared)

    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class", return_value=object
    ):
        _run()

    custom_handlers = setup_logging.call_args.kwargs["custom_handlers"]
    assert custom_handlers["discord"]["webhook_url"] == "https://from-secret"


def test_a_notifier_that_cannot_load_does_not_stop_the_cycle(
    read_bot_config, setup_logging, run_livebot, bot_config, caplog
):
    declared = {
        **DECLARED,
        "live": {**DECLARED["live"], "notifier": {"nosuchplugin": {"log": {}}}},
    }
    read_bot_config.return_value = BotConfig.model_validate(declared)

    with caplog.at_level(logging.ERROR):
        _run()

    run_livebot.assert_called_once()
    assert setup_logging.call_args.kwargs["custom_handlers"] == {}
    assert "Running without the `nosuchplugin` notifier" in caplog.text


def test_a_completed_cycle(run_livebot):
    exit_code = _run()

    assert exit_code == 0


def test_a_config_already_trading_in_another_process(run_livebot, caplog, tmp_path):
    with hold_instance_lock(Path("bot-example.toml")), caplog.at_level(logging.WARNING):
        exit_code = _run()

    assert exit_code == EX_TEMPFAIL
    run_livebot.assert_not_called()
    assert "bot-example.toml is already trading in another process" in caplog.text
    assert (tmp_path / ".rtlab" / "bot-example.lock").is_file()


def test_the_next_cycle_after_a_crash(run_livebot):
    run_livebot.side_effect = [ExchangeCriticalError("the venue rejected us"), None]

    with pytest.raises(ExchangeCriticalError):
        _run()
    _run()

    assert run_livebot.call_count == 2


def _run(**overrides: Any) -> int:
    arguments: dict[str, Any] = {
        "config": Path("bot-example.toml"),
        "logfile_override": None,
        "console": False,
        "logfiles_retention_days": None,
    }
    return main(**{**arguments, **overrides})
