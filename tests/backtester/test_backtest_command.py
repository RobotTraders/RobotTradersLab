import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

import robottraderslab.backtester.command
from robottraderslab.backtester.command import main
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import ExchangeCriticalError, StrategyCriticalError

DECLARED = {
    "strategy": {"strategy_class": "fake-strategy-99"},
    "backtest": {
        "initial_balance": {"USDT": 1000},
        "maker_fee_rate": 0.0002,
        "taker_fee_rate": 0.0006,
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
        "ohlcv_provider": {"name": "ccxt"},
    },
}


@pytest.fixture
def bot_config() -> BotConfig:
    return BotConfig.model_validate(DECLARED)


@pytest.fixture(autouse=True)
def setup_logging() -> Iterator[Mock]:
    with patch(
        "robottraderslab.backtester.command.setup_backtest_logging", autospec=True
    ) as configured:
        yield configured


@pytest.fixture(autouse=True)
def read_bot_config(bot_config: BotConfig) -> Iterator[Mock]:
    with patch.object(
        robottraderslab.backtester.command.BotConfig, "from_file", autospec=True
    ) as loaded:
        loaded.return_value = bot_config
        yield loaded


@pytest.fixture(autouse=True)
def run_backtest() -> Iterator[Mock]:
    with patch("robottraderslab.backtester.command.run_backtest", autospec=True) as ran:
        yield ran


def test_neither_the_cli_nor_the_config_names_a_logfile(setup_logging):
    _run()

    setup_logging.assert_called_once_with(
        logfile=None,
        log_name="bot-example",
        console_level=logging.getLevelName(logging.INFO),
        file_level=logging.getLevelName(logging.INFO),
        enable_console=False,
        enable_file=True,
        custom_handlers={},
        config_dir=None,
    )


def test_the_default_log_is_named_after_the_bot_file(setup_logging):
    _run(config=Path("workspace/pilot/impulse-bitget.toml"))

    assert setup_logging.call_args.kwargs["log_name"] == "impulse-bitget"


def test_the_given_logfile_wins_over_the_config_file(setup_logging):
    _run(logfile_override=Path("my-logfile.log"))

    setup_logging.assert_called_once()
    assert setup_logging.call_args.kwargs["logfile"] == Path("my-logfile.log")


def test_the_config_comes_from_the_given_file(read_bot_config):
    _run(config=Path("my-filename.toml"))

    read_bot_config.assert_called_once_with(Path("my-filename.toml"))


def test_run_backtest_is_called_with_the_loaded_config(bot_config, run_backtest):
    _run()

    run_backtest.assert_called_once_with(bot_config)


def test_a_config_file_that_cannot_be_read(read_bot_config, run_backtest):
    read_bot_config.side_effect = StrategyCriticalError("no config file at `nope`")

    with pytest.raises(StrategyCriticalError, match="no config file"):
        _run()

    run_backtest.assert_not_called()


def test_a_log_notifier_secret_is_resolved_before_logging_setup(
    read_bot_config, setup_logging, tmp_path
):
    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(
        '[[secrets]]\nname = "discord_main"\nwebhook_url = "https://from-secret"\n'
    )
    declared = {
        **DECLARED,
        "secrets_file": str(secrets_file),
        "backtest": {
            **DECLARED["backtest"],
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


def test_the_analyser_measures_the_run_that_just_finished(run_backtest):
    _run()

    run_backtest.return_value.create_analyser.assert_called_once_with(save=True)


def test_no_save_reaches_create_analyser(run_backtest):
    _run(save=False)

    run_backtest.return_value.create_analyser.assert_called_once_with(save=False)


def test_the_chart_is_drawn_for_the_configured_strategy(run_backtest):
    analyser = run_backtest.return_value.create_analyser.return_value

    _run()

    analyser.plot_candlesticks.assert_called_once_with(
        indicators_name="fake-strategy-99"
    )


def test_no_chart_is_drawn_when_asked(run_backtest):
    analyser = run_backtest.return_value.create_analyser.return_value

    _run(chart=False)

    analyser.plot_candlesticks.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        StrategyCriticalError("no indicators"),
        ExchangeCriticalError("no market data"),
        RuntimeError("renderer crashed"),
    ],
    ids=["no_indicators", "venue_unusable", "renderer_error"],
)
def test_a_failure_while_charting_does_not_fail_the_run(run_backtest, caplog, failure):
    analyser = run_backtest.return_value.create_analyser.return_value
    analyser.plot_candlesticks.side_effect = failure

    with caplog.at_level(logging.WARNING):
        _run()

    assert "No chart drawn" in caplog.text


def _run(**overrides: Any) -> None:
    arguments: dict[str, Any] = {
        "config": Path("bot-example.toml"),
        "logfile_override": None,
        "console": False,
        "chart": True,
        "save": True,
    }
    main(**{**arguments, **overrides})
