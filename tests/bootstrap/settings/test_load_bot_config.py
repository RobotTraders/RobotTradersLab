import logging
from pathlib import Path

import pytest

from robottraderslab._core import PlacementReserve
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import StrategyCriticalError


@pytest.fixture
def toml_string() -> str:
    return """\
[strategy]
strategy_class = "dummy"
lookback = 5432
"""


@pytest.fixture
def toml_filename() -> str:
    return str(Path(__file__)).replace(".py", ".toml")


@pytest.fixture
def backtest_preamble() -> str:
    return """\
[strategy]
strategy_class = "dummy"

[backtest]
maker_fee_rate = 0.0
taker_fee_rate = 0.0
start_date = "2024-01-01"
end_date = "2024-01-02"
initial_balance = { USDT = 1000.0 }
"""


@pytest.fixture
def live_preamble() -> str:
    return """\
[strategy]
strategy_class = "dummy"

[live]
trading_account = {}
ohlcv_provider = {}
"""


def test_load_from_file(toml_filename):
    bot_config = BotConfig.from_file(toml_filename)
    assert bot_config.strategy.lookback == 5432


def test_load_from_string(toml_string):
    bot_config = BotConfig.from_text(toml_string)
    assert bot_config.strategy.lookback == 5432


def test_from_string_updates_from_file(toml_filename):
    overlay = """\
[strategy]
lookback = 2345
updated = true
"""

    bot_config = BotConfig.from_file(toml_filename, overlay=overlay)

    assert bot_config.strategy.strategy_class == "dummy"
    assert bot_config.strategy.lookback == 2345
    assert bot_config.strategy.updated


def test_relative_storage_dir_resolved_against_toml_directory(
    tmp_path, backtest_preamble
):
    toml_file = tmp_path / "workspace" / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir(parents=True)
    toml_file.write_text(
        backtest_preamble + 'ohlcv_provider = { storage_dir = "../data/ohlcvs" }\n'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    resolved = bot_config.backtest.ohlcv_provider["storage_dir"]
    assert Path(resolved).is_absolute()
    assert resolved == str(toml_file.parent / ".." / "data" / "ohlcvs")


def test_relative_secrets_file_resolved_against_toml_directory(tmp_path):
    toml_file = tmp_path / "workspace" / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir(parents=True)
    toml_file.write_text(
        'secrets_file = "secrets.toml"\n\n[strategy]\nstrategy_class = "dummy"\n'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    resolved = bot_config.secrets_file
    assert Path(resolved).is_absolute()
    assert resolved == str(toml_file.parent / "secrets.toml")


def test_relative_file_resolved_against_toml_directory(tmp_path, backtest_preamble):
    toml_file = tmp_path / "workspace" / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir(parents=True)
    toml_file.write_text(
        backtest_preamble + 'ohlcv_provider = { file = "../data/btc.csv" }\n'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    resolved = bot_config.backtest.ohlcv_provider["file"]
    assert resolved == str(toml_file.parent / ".." / "data" / "btc.csv")


def test_relative_dir_of_a_storage_table_resolved_against_toml_directory(
    tmp_path, backtest_preamble
):
    toml_file = tmp_path / "workspace" / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir(parents=True)
    toml_file.write_text(
        backtest_preamble
        + 'ohlcv_provider = { storage = { type = "csv", dir = "../data/ohlcvs" } }\n'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    resolved = bot_config.backtest.ohlcv_provider["storage"]["dir"]
    assert resolved == str(toml_file.parent / ".." / "data" / "ohlcvs")


def test_relative_live_file_resolved_against_toml_directory(tmp_path, live_preamble):
    toml_file = tmp_path / "workspace" / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir(parents=True)
    toml_file.write_text(
        live_preamble.replace(
            "ohlcv_provider = {}", 'ohlcv_provider = { file = "../data/btc.csv" }'
        )
    )

    bot_config = BotConfig.from_file(str(toml_file))

    resolved = bot_config.live.ohlcv_provider["file"]
    assert resolved == str(toml_file.parent / ".." / "data" / "btc.csv")


def test_relative_live_dir_of_a_storage_table_resolved_against_toml_directory(
    tmp_path, live_preamble
):
    toml_file = tmp_path / "workspace" / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir(parents=True)
    toml_file.write_text(
        live_preamble.replace(
            "ohlcv_provider = {}",
            'ohlcv_provider = { storage = { type = "csv", dir = "../data/ohlcvs" } }',
        )
    )

    bot_config = BotConfig.from_file(str(toml_file))

    resolved = bot_config.live.ohlcv_provider["storage"]["dir"]
    assert resolved == str(toml_file.parent / ".." / "data" / "ohlcvs")


def test_a_bot_naming_no_store_takes_its_workspace_root(tmp_path, backtest_preamble):
    workspace = tmp_path / "workspace"
    (workspace / ".rtlab").mkdir(parents=True)
    toml_file = workspace / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir()
    toml_file.write_text(
        backtest_preamble + 'ohlcv_provider = { ohlcv_provider = "ccxt_bitget" }'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    stored_at = bot_config.backtest.ohlcv_provider["storage_dir"]
    assert stored_at == str(workspace / "data" / "ohlcvs")


def test_a_bot_naming_no_store_outside_a_marked_workspace(tmp_path, backtest_preamble):
    toml_file = tmp_path / "workspace" / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir(parents=True)
    toml_file.write_text(
        backtest_preamble + 'ohlcv_provider = { ohlcv_provider = "ccxt_bitget" }'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    stored_at = bot_config.backtest.ohlcv_provider["storage_dir"]
    assert stored_at == str(toml_file.parent / "data" / "ohlcvs")


def test_a_live_bot_naming_no_store_takes_its_workspace_root(tmp_path, live_preamble):
    workspace = tmp_path / "workspace"
    (workspace / ".rtlab").mkdir(parents=True)
    toml_file = workspace / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir()
    toml_file.write_text(
        live_preamble.replace(
            "ohlcv_provider = {}", 'ohlcv_provider = { ohlcv_provider = "bitget" }'
        )
    )

    bot_config = BotConfig.from_file(str(toml_file))

    stored_at = bot_config.live.ohlcv_provider["storage_dir"]
    assert stored_at == str(workspace / "data" / "ohlcvs")


def test_a_source_reading_no_venue_takes_no_store(tmp_path, backtest_preamble):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        backtest_preamble
        + 'ohlcv_provider = { ohlcv_provider = "csv", file = "btc.csv", symbol = "BTC/USDT:USDT", timeframe = "15m" }'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    assert "storage_dir" not in bot_config.backtest.ohlcv_provider


def test_a_generated_source_takes_no_store(tmp_path, backtest_preamble):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        backtest_preamble + 'ohlcv_provider = { ohlcv_provider = "mock" }'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    assert "storage_dir" not in bot_config.backtest.ohlcv_provider


def test_absolute_storage_dir_unchanged(tmp_path, backtest_preamble):
    absolute_path = (tmp_path / "data" / "ohlcvs").as_posix()
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        backtest_preamble + f'ohlcv_provider = {{ storage_dir = "{absolute_path}" }}\n'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    assert bot_config.backtest.ohlcv_provider["storage_dir"] == absolute_path


def test_a_backtest_naming_no_fee_convention(backtest_preamble):
    bot_config = BotConfig.from_text(
        backtest_preamble
        + """ohlcv_provider = {}
"""
    )

    assert bot_config.backtest.fee_mode == "exchange"
    assert bot_config.backtest.placement_reserve == PlacementReserve()


def test_a_backtest_naming_the_exchange_convention_and_its_reserve(backtest_preamble):
    bot_config = BotConfig.from_text(
        backtest_preamble
        + """ohlcv_provider = {}
fee_mode = "exchange"

[backtest.placement_reserve]
margin_markup = 0.01
"""
    )

    assert bot_config.backtest.fee_mode == "exchange"
    assert bot_config.backtest.placement_reserve.margin_markup == 0.01


def test_config_dir_set_to_toml_parent(tmp_path):
    toml_file = tmp_path / "workspace" / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir(parents=True)
    toml_file.write_text("[strategy]\nstrategy_class = 'dummy'\n")

    bot_config = BotConfig.from_file(str(toml_file))

    assert bot_config.config_dir == toml_file.resolve().parent


def test_relative_reports_dir_resolved_against_toml_directory(
    tmp_path, backtest_preamble
):
    toml_file = tmp_path / "workspace" / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir(parents=True)
    toml_file.write_text(
        backtest_preamble
        + 'ohlcv_provider = {}\n\n[report]\nreports_dir = "my_reports"\n'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    assert bot_config.report.reports_dir == toml_file.parent / "my_reports"


def test_relative_backtest_logfile_resolved_against_toml_directory(
    tmp_path, backtest_preamble
):
    toml_file = tmp_path / "workspace" / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir(parents=True)
    toml_file.write_text(
        backtest_preamble
        + 'ohlcv_provider = {}\n\n[backtest.logging]\nlogfile = "logs/backtest.log"\n'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    resolved = bot_config.backtest.logging.logfile
    assert Path(resolved).is_absolute()
    assert resolved == str(toml_file.parent / "logs" / "backtest.log")


def test_relative_live_logfile_resolved_against_toml_directory(tmp_path, live_preamble):
    toml_file = tmp_path / "workspace" / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir(parents=True)
    toml_file.write_text(
        live_preamble + '\n[live.logging]\nlogfile = "logs/live.log"\n'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    resolved = bot_config.live.logging.logfile
    assert Path(resolved).is_absolute()
    assert resolved == str(toml_file.parent / "logs" / "live.log")


def test_secrets_file_discovered_beside_the_config(tmp_path):
    workspace = tmp_path / "workspace" / "my_strategy"
    workspace.mkdir(parents=True)
    toml_file = workspace / "bot-example.toml"
    toml_file.write_text('[strategy]\nstrategy_class = "dummy"\n')
    (workspace / "secrets.toml").write_text("")

    bot_config = BotConfig.from_file(str(toml_file))

    assert bot_config.secrets_file == str(workspace / "secrets.toml")


def test_secrets_file_discovered_in_the_workspace_root(tmp_path):
    workspace_root = tmp_path / "workspace"
    bot_dir = workspace_root / "my_strategy"
    bot_dir.mkdir(parents=True)
    toml_file = bot_dir / "bot-example.toml"
    toml_file.write_text('[strategy]\nstrategy_class = "dummy"\n')
    (workspace_root / "secrets.toml").write_text("")

    bot_config = BotConfig.from_file(str(toml_file))

    assert bot_config.secrets_file == str(workspace_root / "secrets.toml")


def test_secrets_file_discovery_prefers_the_bot_local_file(tmp_path):
    workspace_root = tmp_path / "workspace"
    bot_dir = workspace_root / "my_strategy"
    bot_dir.mkdir(parents=True)
    toml_file = bot_dir / "bot-example.toml"
    toml_file.write_text('[strategy]\nstrategy_class = "dummy"\n')
    (bot_dir / "secrets.toml").write_text("")
    (workspace_root / "secrets.toml").write_text("")

    bot_config = BotConfig.from_file(str(toml_file))

    assert bot_config.secrets_file == str(bot_dir / "secrets.toml")


def test_no_secrets_file_discovered_anywhere_leaves_it_unset(tmp_path):
    toml_file = tmp_path / "workspace" / "my_strategy" / "bot-example.toml"
    toml_file.parent.mkdir(parents=True)
    toml_file.write_text('[strategy]\nstrategy_class = "dummy"\n')

    bot_config = BotConfig.from_file(str(toml_file))

    assert bot_config.secrets_file is None


def test_an_explicit_secrets_file_is_not_replaced_by_discovery(tmp_path):
    workspace = tmp_path / "workspace" / "my_strategy"
    workspace.mkdir(parents=True)
    toml_file = workspace / "bot-example.toml"
    toml_file.write_text(
        'secrets_file = "declared.toml"\n\n[strategy]\nstrategy_class = "dummy"\n'
    )
    (workspace / "secrets.toml").write_text("")

    bot_config = BotConfig.from_file(str(toml_file))

    assert bot_config.secrets_file == str(workspace / "declared.toml")


def test_absolute_secrets_file_unchanged(tmp_path):
    absolute_path = (tmp_path / "secrets.toml").as_posix()
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        f'secrets_file = "{absolute_path}"\n\n[strategy]\nstrategy_class = "dummy"\n'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    assert bot_config.secrets_file == absolute_path


def test_missing_paths_do_not_cause_errors(tmp_path):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text('[strategy]\nstrategy_class = "dummy"\n')

    bot_config = BotConfig.from_file(str(toml_file))

    assert bot_config.secrets_file is None
    assert bot_config.backtest is None
    assert bot_config.live is None


def test_config_dir_not_set_when_loading_from_string_only(toml_string):
    bot_config = BotConfig.from_text(toml_string)

    assert bot_config.config_dir is None


def test_config_file_set_to_the_file_it_was_loaded_from(toml_filename):
    bot_config = BotConfig.from_file(toml_filename)

    assert bot_config.config_file == Path(toml_filename)


def test_config_file_not_set_when_loading_from_string_only(toml_string):
    bot_config = BotConfig.from_text(toml_string)

    assert bot_config.config_file is None


def test_the_environment_cannot_supply_a_declared_field(tmp_path, monkeypatch):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text('[strategy]\nstrategy_class = "dummy"\n')
    monkeypatch.setenv("SECRETS_FILE", "leaked.toml")

    bot_config = BotConfig.from_file(str(toml_file))

    assert bot_config.secrets_file is None


def test_the_environment_cannot_replace_a_declared_field(tmp_path, monkeypatch):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        """\
secrets_file = "declared.toml"

[strategy]
strategy_class = "dummy"
"""
    )
    monkeypatch.setenv("SECRETS_FILE", "leaked.toml")

    bot_config = BotConfig.from_file(str(toml_file))

    assert bot_config.secrets_file == str(toml_file.parent / "declared.toml")


def test_the_retired_flat_notifiers_table(tmp_path):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        """\
[strategy]
strategy_class = "dummy"

[live]
ohlcv_provider = {}
trading_account = {}
notifiers = { discord = { webhook_url = "https://example.invalid" } }
"""
    )

    with pytest.raises(StrategyCriticalError, match="declare each subscription under"):
        BotConfig.from_file(str(toml_file))


def test_the_retired_custom_handlers_table(tmp_path):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        """\
[strategy]
strategy_class = "dummy"

[live]
ohlcv_provider = {}
trading_account = {}

[live.logging]
custom_handlers = { discord = { class = "x" } }
"""
    )

    with pytest.raises(StrategyCriticalError, match="declare the handler under"):
        BotConfig.from_file(str(toml_file))


def test_the_retired_backtest_reference_symbol(tmp_path, backtest_preamble):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        backtest_preamble + 'ohlcv_provider = {}\nreference_symbol = "BTC/USDT:USDT"\n'
    )

    with pytest.raises(StrategyCriticalError, match=r"declare it under `\[report\]`"):
        BotConfig.from_file(str(toml_file))


def test_the_retired_backtest_reports_dir(tmp_path, backtest_preamble):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        backtest_preamble + 'ohlcv_provider = {}\nreports_dir = "reports"\n'
    )

    with pytest.raises(
        StrategyCriticalError, match=r"`backtest.reports_dir` is not read"
    ):
        BotConfig.from_file(str(toml_file))


def test_a_reference_symbol_the_engine_cannot_parse(tmp_path):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        '[strategy]\nstrategy_class = "dummy"\n\n[report]\nreference_symbol = "BTC"\n'
    )

    with pytest.raises(StrategyCriticalError, match="Invalid symbol format: BTC"):
        BotConfig.from_file(str(toml_file))


def test_a_reporting_key_the_engine_does_not_declare(tmp_path):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        '[strategy]\nstrategy_class = "dummy"\n\n[report]\nreference_symbo = "BTC/USDT"\n'
    )

    with pytest.raises(StrategyCriticalError, match="reference_symbo"):
        BotConfig.from_file(str(toml_file))


def test_a_section_the_engine_does_not_declare(tmp_path):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        '[strategy]\nstrategy_class = "dummy"\n\n[bakctest]\nfee = 0.1\n'
    )

    with pytest.raises(StrategyCriticalError, match="bakctest"):
        BotConfig.from_file(str(toml_file))


def test_a_bot_declaring_no_strategy(tmp_path):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text("[live]\n")

    with pytest.raises(StrategyCriticalError, match="strategy"):
        BotConfig.from_file(str(toml_file))


def test_a_config_file_that_is_not_there(tmp_path):
    missing = tmp_path / "absent" / "bot-example.toml"

    with pytest.raises(StrategyCriticalError, match=str(missing.name)):
        BotConfig.from_file(str(missing))


def test_a_credential_written_in_the_file_is_never_logged(tmp_path, caplog):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        '[strategy]\nstrategy_class = "dummy"\n\n[live]\n'
        'ohlcv_provider = {}\ntrading_account = { exchange = "bitget", '
        'api_key = "a-real-key" }\n'
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(StrategyCriticalError):
        BotConfig.from_file(str(toml_file))

    assert "a-real-key" not in caplog.text


def test_an_inline_credential_in_trading_account_is_refused(tmp_path):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        '[strategy]\nstrategy_class = "dummy"\n\n[live]\n'
        'ohlcv_provider = {}\ntrading_account = { exchange = "bitget", '
        'secret_key = "shh" }\n'
    )

    with pytest.raises(StrategyCriticalError, match="live.trading_account.secret_key"):
        BotConfig.from_file(str(toml_file))


def test_an_inline_credential_in_a_live_notifier_occasion_is_refused(tmp_path):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        '[strategy]\nstrategy_class = "dummy"\n\n[live]\n'
        "ohlcv_provider = {}\ntrading_account = {}\n\n"
        '[live.notifier.discord.fills]\nwebhook_url = "https://example.invalid"\n'
    )

    with pytest.raises(
        StrategyCriticalError, match="live.notifier.discord.fills.webhook_url"
    ):
        BotConfig.from_file(str(toml_file))


def test_an_inline_credential_in_a_backtest_notifier_occasion_is_refused(
    tmp_path, backtest_preamble
):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        backtest_preamble + "ohlcv_provider = {}\n\n"
        '[backtest.notifier.discord.log]\nwebhook_url = "https://example.invalid"\n'
    )

    with pytest.raises(
        StrategyCriticalError, match="backtest.notifier.discord.log.webhook_url"
    ):
        BotConfig.from_file(str(toml_file))


def test_a_secret_name_alone_loads_the_account_as_today(tmp_path):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        '[strategy]\nstrategy_class = "dummy"\n\n[live]\n'
        'ohlcv_provider = {}\ntrading_account = { exchange = "bitget", '
        'secret_name = "demoaccount" }\n'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    assert bot_config.live.trading_account == {
        "exchange": "bitget",
        "secret_name": "demoaccount",
    }


def test_a_secret_name_alone_loads_the_notifier_as_today(tmp_path):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text(
        '[strategy]\nstrategy_class = "dummy"\n\n[live]\n'
        "ohlcv_provider = {}\ntrading_account = {}\n\n"
        '[live.notifier.discord.fills]\nsecret_name = "discord_notifs"\n'
    )

    bot_config = BotConfig.from_file(str(toml_file))

    assert bot_config.live.notifier == {
        "discord": {"fills": {"secret_name": "discord_notifs"}}
    }


def test_a_config_that_is_not_valid_toml_names_the_file(tmp_path):
    toml_file = tmp_path / "bot-example.toml"
    toml_file.write_text("[[bot]\n")

    with pytest.raises(StrategyCriticalError, match="is not valid TOML"):
        BotConfig.from_file(str(toml_file))
