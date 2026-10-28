import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

import robottraderslab.backtester.command
from robottraderslab.backtester.command import main
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import BotCriticalError, ExchangeCriticalError

DECLARED = {
    "name": "test-bot",
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
def bot_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "test-bot"
    folder.mkdir()
    (folder / "bot.toml").write_text("")
    return folder


@pytest.fixture
def bot_config(bot_folder: Path) -> BotConfig:
    return BotConfig.model_validate({**DECLARED, "config_dir": bot_folder})


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


@pytest.fixture
def analyser(run_backtest: Mock) -> Mock:
    return run_backtest.return_value.create_analyser.return_value


def test_neither_the_cli_nor_the_config_names_a_logfile(setup_logging, bot_folder):
    _run(bot_folder)

    logfile = setup_logging.call_args.kwargs["logfile"]
    assert logfile.name == "backtest.log"
    assert logfile.parent.parent == bot_folder / "backtests"
    assert setup_logging.call_args.kwargs["enable_console"] is False


def test_the_saved_reports_complete_the_folder_the_log_opened(
    setup_logging, analyser, bot_config, bot_folder
):
    _run(bot_folder)

    logfile = setup_logging.call_args.kwargs["logfile"]
    timestamp = analyser.save_run.call_args.args[2]
    assert logfile.parent.name.startswith(timestamp)
    analyser.save_run.assert_called_once_with(
        bot_config, "fake-strategy-99", timestamp, "2024-01-01"
    )


def test_the_given_logfile_wins_over_the_run_folder(setup_logging, bot_folder):
    _run(bot_folder, logfile_override=Path("my-logfile.log"))

    assert setup_logging.call_args.kwargs["logfile"] == Path("my-logfile.log")


def test_the_config_logfile_wins_over_the_run_folder(
    read_bot_config, setup_logging, bot_folder
):
    read_bot_config.return_value = _with_logging(
        bot_folder, {"logfile": "my-logfile.log"}
    )

    _run(bot_folder)

    assert setup_logging.call_args.kwargs["logfile"] == Path("my-logfile.log")


def test_the_given_logfile_wins_over_the_config_logfile(
    read_bot_config, setup_logging, bot_folder
):
    read_bot_config.return_value = _with_logging(
        bot_folder, {"logfile": "configured.log"}
    )

    _run(bot_folder, logfile_override=Path("given.log"))

    assert setup_logging.call_args.kwargs["logfile"] == Path("given.log")


def test_an_unsaved_run_logs_to_the_terminal_alone(setup_logging, bot_folder):
    _run(bot_folder, save=False)

    assert setup_logging.call_args.kwargs["logfile"] is None
    assert setup_logging.call_args.kwargs["enable_console"] is True


def test_an_unsaved_run_still_writes_the_given_logfile(setup_logging, bot_folder):
    _run(bot_folder, save=False, logfile_override=Path("my-logfile.log"))

    assert setup_logging.call_args.kwargs["logfile"] == Path("my-logfile.log")
    assert setup_logging.call_args.kwargs["enable_console"] is False


def test_a_disabled_log_file_shows_in_the_terminal(
    read_bot_config, setup_logging, bot_folder
):
    read_bot_config.return_value = _with_logging(bot_folder, {"enable_file": False})

    _run(bot_folder, logfile_override=Path("my-logfile.log"))

    assert setup_logging.call_args.kwargs["logfile"] is None
    assert setup_logging.call_args.kwargs["enable_console"] is True


def test_the_config_comes_from_the_bot_toml_of_the_given_folder(
    read_bot_config, bot_folder
):
    _run(bot_folder)

    read_bot_config.assert_called_once_with(bot_folder / "bot.toml")


def test_run_backtest_is_called_with_the_loaded_config(
    bot_config, run_backtest, bot_folder
):
    _run(bot_folder)

    run_backtest.assert_called_once_with(bot_config, record_log=False)


def test_a_config_file_that_cannot_be_read(read_bot_config, run_backtest, bot_folder):
    read_bot_config.side_effect = BotCriticalError("no config file at `nope`")

    with pytest.raises(BotCriticalError, match="no config file"):
        _run(bot_folder)

    run_backtest.assert_not_called()


def test_a_log_notifier_secret_is_resolved_before_logging_setup(
    read_bot_config, setup_logging, tmp_path, bot_folder
):
    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(
        '[[secrets]]\nname = "discord_main"\nwebhook_url = "https://from-secret"\n'
    )
    declared = {
        **DECLARED,
        "config_dir": bot_folder,
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
        _run(bot_folder)

    custom_handlers = setup_logging.call_args.kwargs["custom_handlers"]
    assert custom_handlers["discord"]["webhook_url"] == "https://from-secret"


def test_no_save_saves_nothing(analyser, bot_folder):
    _run(bot_folder, save=False)

    analyser.save_run.assert_not_called()


def test_a_run_that_closed_no_trade_is_refused(analyser, bot_folder):
    analyser.closed_trades_count = 0

    with pytest.raises(RuntimeError, match="No trades generated by the backtest"):
        _run(bot_folder)

    analyser.save_run.assert_not_called()


def test_the_chart_is_drawn_for_the_configured_strategy(analyser, bot_folder):
    _run(bot_folder)

    analyser.plot_candlesticks.assert_called_once_with(
        indicators_name="fake-strategy-99"
    )


def test_no_chart_is_drawn_when_asked(analyser, bot_folder):
    _run(bot_folder, chart=False)

    analyser.plot_candlesticks.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        BotCriticalError("no indicators"),
        ExchangeCriticalError("no market data"),
        RuntimeError("renderer crashed"),
    ],
    ids=["no_indicators", "venue_unusable", "renderer_error"],
)
def test_a_failure_while_charting_does_not_fail_the_run(
    bot_folder, analyser, caplog, failure
):
    analyser.plot_candlesticks.side_effect = failure

    with caplog.at_level(logging.WARNING):
        _run(bot_folder)

    assert "No chart drawn" in caplog.text


def _run(folder: Path, **overrides: Any) -> None:
    arguments: dict[str, Any] = {
        "folder": folder,
        "logfile_override": None,
        "console": False,
        "chart": True,
        "save": True,
    }
    main(**{**arguments, **overrides})


def _with_logging(bot_folder: Path, logging_keys: dict[str, Any]) -> BotConfig:
    declared = {
        **DECLARED,
        "config_dir": bot_folder,
        "backtest": {**DECLARED["backtest"], "logging": logging_keys},
    }
    return BotConfig.model_validate(declared)
