import re

import pytest

from robottraderslab.bootstrap import load_ohlcv_provider, require_ohlcv_provider
from robottraderslab.exceptions import StrategyCriticalError


def test_load_mock_provider():
    settings = {"ohlcv_provider": "mock", "lookback": 100}
    provider = load_ohlcv_provider(settings)
    assert provider.__class__.__name__ == "MockOHLCVProvider"


def test_load_csv_provider(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "date,open,high,low,close,volume\n2024-01-01,100,105,95,102,1000\n"
    )
    settings = {
        "ohlcv_provider": "csv",
        "file": str(csv_file),
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1d",
    }
    provider = load_ohlcv_provider(settings)
    assert provider.__class__.__name__ == "CSVOHLCVProvider"


def test_load_exchange_provider(tmp_path, stub_exchange):
    settings = {"ohlcv_provider": stub_exchange, "storage_dir": str(tmp_path)}
    provider = load_ohlcv_provider(settings)
    assert provider.__class__.__name__ == "ExchangeOHLCVProvider"


def test_load_ccxt_provider(tmp_path):
    settings = {"ohlcv_provider": "ccxt_binance", "storage_dir": str(tmp_path)}
    provider = load_ohlcv_provider(settings)
    assert provider.__class__.__name__ == "ExchangeOHLCVProvider"


def test_case_insensitive():
    settings = {"ohlcv_provider": "MOCK", "lookback": 100}
    provider = load_ohlcv_provider(settings)
    assert provider.__class__.__name__ == "MockOHLCVProvider"


def test_exchange_provider_accepts_nested_storage(tmp_path, stub_exchange):
    settings = {
        "ohlcv_provider": stub_exchange,
        "storage": {"type": "csv", "dir": str(tmp_path)},
    }
    provider = load_ohlcv_provider(settings)
    assert provider.__class__.__name__ == "ExchangeOHLCVProvider"


def test_an_adapter_setting_reaches_the_provider(tmp_path, stub_exchange):
    settings = {"ohlcv_provider": stub_exchange, "storage_dir": str(tmp_path)}
    provider = load_ohlcv_provider(settings, variant="demo")
    assert provider._dataset_name == f"{stub_exchange}-demo"


def test_a_csv_source_missing_its_symbol_and_timeframe():
    with pytest.raises(StrategyCriticalError, match="needs `symbol`, `timeframe`"):
        require_ohlcv_provider(
            {"ohlcv_provider": "csv", "file": "btc.csv"},
            "backtest.ohlcv_provider",
            None,
        )


def test_a_mock_source_given_a_key_it_does_not_take():
    with pytest.raises(StrategyCriticalError, match="does not take `seed`"):
        require_ohlcv_provider(
            {"ohlcv_provider": "mock", "seed": 1}, "backtest.ohlcv_provider", None
        )


def test_an_exchange_sources_own_key_misspelt():
    with pytest.raises(StrategyCriticalError, match="does not take `storage_dirr`"):
        require_ohlcv_provider(
            {"ohlcv_provider": "ccxt_bitget", "storage_dirr": "data"},
            "live.ohlcv_provider",
            None,
        )


def test_an_exchange_sources_adapter_key_misspelt(stub_exchange):
    with pytest.raises(StrategyCriticalError, match="does not take `variantt`"):
        require_ohlcv_provider(
            {"ohlcv_provider": stub_exchange, "variantt": "demo"},
            "live.ohlcv_provider",
            None,
        )


def test_an_oanda_sources_adapter_key_misspelt():
    pytest.importorskip("robottraderslab_oanda")
    with pytest.raises(StrategyCriticalError, match="does not take `access_tokenn`"):
        require_ohlcv_provider(
            {"ohlcv_provider": "oanda", "access_tokenn": "token"},
            "live.ohlcv_provider",
            None,
        )


def test_a_dukascopy_sources_adapter_key_misspelt():
    pytest.importorskip("robottraderslab_dukascopy")
    with pytest.raises(StrategyCriticalError, match="does not take `offer_sde`"):
        require_ohlcv_provider(
            {"ohlcv_provider": "dukascopy", "offer_sde": "ask"},
            "live.ohlcv_provider",
            None,
        )


def test_an_oanda_source_missing_the_keys_its_adapter_needs():
    pytest.importorskip("robottraderslab_oanda")
    with pytest.raises(
        StrategyCriticalError, match="needs `access_token`, `account_id`"
    ):
        require_ohlcv_provider({"ohlcv_provider": "oanda"}, "live.ohlcv_provider", None)


def test_a_ccxt_source_carries_no_venue_key_of_its_own(tmp_path):
    require_ohlcv_provider(
        {"ohlcv_provider": "ccxt_bitget", "storage_dir": str(tmp_path)},
        "live.ohlcv_provider",
        None,
    )


def test_a_ccxt_source_writing_its_venue_is_refused():
    with pytest.raises(StrategyCriticalError, match="does not take `exchange_name`"):
        require_ohlcv_provider(
            {"ohlcv_provider": "ccxt_bitget", "exchange_name": "binance"},
            "live.ohlcv_provider",
            None,
        )


def test_a_csv_source_both_missing_and_unknown_keys():
    with pytest.raises(
        StrategyCriticalError,
        match="needs `symbol`, `timeframe` and does not take `symbol_typo`",
    ):
        require_ohlcv_provider(
            {
                "ohlcv_provider": "csv",
                "file": "btc.csv",
                "symbol_typo": "BTC/USDT:USDT",
            },
            "backtest.ohlcv_provider",
            None,
        )


def test_the_settings_file_is_named_in_the_message(tmp_path):
    config_file = tmp_path / "bot.toml"

    with pytest.raises(StrategyCriticalError, match=re.escape(str(config_file))):
        require_ohlcv_provider(
            {"ohlcv_provider": "mock", "seed": 1},
            "backtest.ohlcv_provider",
            config_file,
        )


def test_a_clean_csv_source_is_accepted():
    require_ohlcv_provider(
        {
            "ohlcv_provider": "csv",
            "file": "btc.csv",
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1d",
        },
        "backtest.ohlcv_provider",
        None,
    )


def test_a_clean_mock_source_is_accepted():
    require_ohlcv_provider(
        {"ohlcv_provider": "mock", "lookback": 100}, "backtest.ohlcv_provider", None
    )


def test_a_clean_exchange_source_is_accepted(tmp_path, stub_exchange):
    require_ohlcv_provider(
        {
            "ohlcv_provider": stub_exchange,
            "storage_dir": str(tmp_path),
            "variant": "demo",
        },
        "live.ohlcv_provider",
        None,
    )


def test_a_clean_oanda_source_is_accepted(tmp_path):
    pytest.importorskip("robottraderslab_oanda")
    require_ohlcv_provider(
        {
            "ohlcv_provider": "oanda",
            "storage_dir": str(tmp_path),
            "access_token": "token",
            "account_id": "101-004-1234567-001",
            "practice": True,
        },
        "live.ohlcv_provider",
        None,
    )


def test_a_clean_dukascopy_source_is_accepted(tmp_path):
    pytest.importorskip("robottraderslab_dukascopy")
    require_ohlcv_provider(
        {
            "ohlcv_provider": "dukascopy",
            "storage_dir": str(tmp_path),
            "offer_side": "ask",
        },
        "live.ohlcv_provider",
        None,
    )


def test_a_table_naming_no_source():
    with pytest.raises(StrategyCriticalError, match="needs `ohlcv_provider`"):
        require_ohlcv_provider({"file": "btc.csv"}, "backtest.ohlcv_provider", None)


def test_an_unresolvable_adapter_is_left_to_fail_later():
    require_ohlcv_provider(
        {"ohlcv_provider": "no_such_adapter", "anything": True},
        "live.ohlcv_provider",
        None,
    )
