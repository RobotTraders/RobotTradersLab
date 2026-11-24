import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from robottraderslab import Symbol
from robottraderslab.ohlcv_provider.exchange.download_metadata import (
    _meta_path,
    load_earliest_available,
    load_empty_ranges,
    store_earliest_available,
)


@pytest.fixture
def symbol() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def storage_dir(tmp_path: Path) -> Path:
    return tmp_path / "ohlcv_storage"


class TestLoadEarliestAvailable:
    def test_load_when_file_does_not_exist(self, storage_dir: Path, symbol: Symbol):
        loaded = load_earliest_available(storage_dir, "binance", symbol, "5m")

        assert loaded is None

    def test_load_with_corrupted_json(self, storage_dir: Path, symbol: Symbol):
        meta_file = _meta_path(storage_dir, "binance", symbol, "5m")
        meta_file.parent.mkdir(parents=True)
        meta_file.write_text("not valid json {{{")

        loaded = load_earliest_available(storage_dir, "binance", symbol, "5m")

        assert loaded is None
        assert not meta_file.exists()

    def test_load_deletes_corrupted_file(self, storage_dir: Path, symbol: Symbol):
        meta_file = _meta_path(storage_dir, "binance", symbol, "5m")
        meta_file.parent.mkdir(parents=True)
        meta_file.write_text("garbage data !@#$%")

        loaded = load_earliest_available(storage_dir, "binance", symbol, "5m")

        assert loaded is None
        assert not meta_file.exists()

    def test_load_with_missing_key(self, storage_dir: Path, symbol: Symbol):
        meta_file = _meta_path(storage_dir, "binance", symbol, "5m")
        meta_file.parent.mkdir(parents=True)
        meta_file.write_text(json.dumps({"some_other_key": "value"}))

        loaded = load_earliest_available(storage_dir, "binance", symbol, "5m")

        assert loaded is None

    @pytest.mark.parametrize(
        "stored",
        ["the-big-bang", 1700000000],
        ids=["unparseable_string", "not_a_string"],
    )
    def test_an_unreadable_boundary_names_the_metadata_file(
        self, storage_dir: Path, symbol: Symbol, caplog, stored
    ):
        meta_file = _meta_path(storage_dir, "binance", symbol, "5m")
        meta_file.parent.mkdir(parents=True)
        meta_file.write_text(json.dumps({"earliest_available": stored}))

        with caplog.at_level(logging.WARNING):
            loaded = load_earliest_available(storage_dir, "binance", symbol, "5m")

        assert loaded is None
        assert str(meta_file) in caplog.text

    def test_a_file_without_the_key_is_no_cause_for_a_warning(
        self, storage_dir: Path, symbol: Symbol, caplog
    ):
        meta_file = _meta_path(storage_dir, "binance", symbol, "5m")
        meta_file.parent.mkdir(parents=True)
        meta_file.write_text(json.dumps({"empty_ranges": []}))

        with caplog.at_level(logging.WARNING):
            load_earliest_available(storage_dir, "binance", symbol, "5m")

        assert caplog.text == ""


class TestLoadEmptyRanges:
    def test_an_unreadable_range_names_the_metadata_file(
        self, storage_dir: Path, symbol: Symbol, caplog
    ):
        meta_file = _meta_path(storage_dir, "binance", symbol, "5m")
        meta_file.parent.mkdir(parents=True)
        meta_file.write_text(
            json.dumps(
                {
                    "empty_ranges": [
                        ["2024-01-01T00:00:00", "yesterday"],
                        ["2024-02-01T00:00:00", "2024-02-02T00:00:00"],
                    ]
                }
            )
        )

        with caplog.at_level(logging.WARNING):
            ranges = load_empty_ranges(storage_dir, "binance", symbol, "5m")

        assert ranges == [(datetime(2024, 2, 1), datetime(2024, 2, 2))]
        assert str(meta_file) in caplog.text

    def test_an_unreadable_list_names_the_metadata_file(
        self, storage_dir: Path, symbol: Symbol, caplog
    ):
        meta_file = _meta_path(storage_dir, "binance", symbol, "5m")
        meta_file.parent.mkdir(parents=True)
        meta_file.write_text(json.dumps({"empty_ranges": "all of them"}))

        with caplog.at_level(logging.WARNING):
            ranges = load_empty_ranges(storage_dir, "binance", symbol, "5m")

        assert ranges == []
        assert str(meta_file) in caplog.text

    def test_a_file_without_the_key_is_no_cause_for_a_warning(
        self, storage_dir: Path, symbol: Symbol, caplog
    ):
        meta_file = _meta_path(storage_dir, "binance", symbol, "5m")
        meta_file.parent.mkdir(parents=True)
        meta_file.write_text(json.dumps({"earliest_available": "2024-01-01T00:00:00"}))

        with caplog.at_level(logging.WARNING):
            ranges = load_empty_ranges(storage_dir, "binance", symbol, "5m")

        assert ranges == []
        assert caplog.text == ""


class TestStoreEarliestAvailable:
    def test_store_and_load_round_trip(self, storage_dir: Path, symbol: Symbol):
        earliest = datetime(2023, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        store_earliest_available(storage_dir, "binance", symbol, "5m", earliest)
        loaded = load_earliest_available(storage_dir, "binance", symbol, "5m")

        assert loaded == earliest

    def test_store_creates_parent_directories(self, storage_dir: Path, symbol: Symbol):
        earliest = datetime(2024, 1, 1, tzinfo=timezone.utc)

        assert not storage_dir.exists()

        store_earliest_available(storage_dir, "binance", symbol, "5m", earliest)

        meta_file = _meta_path(storage_dir, "binance", symbol, "5m")
        assert meta_file.exists()


class TestMetaPath:
    def test__meta_path_strips_ccxt_prefix(self, storage_dir: Path, symbol: Symbol):
        path_with_prefix = _meta_path(storage_dir, "ccxt_binance", symbol, "5m")
        path_without_prefix = _meta_path(storage_dir, "binance", symbol, "5m")

        assert path_with_prefix == path_without_prefix
        relative_parts = path_with_prefix.relative_to(storage_dir).parts
        assert all("ccxt_" not in part for part in relative_parts)
        assert "binance" in relative_parts

    def test__meta_path_sanitizes_symbol(self, storage_dir: Path):
        symbol = Symbol.create("BTC/USDT:USDT")

        path = _meta_path(storage_dir, "binance", symbol, "1h")

        assert "/" not in path.name
        assert ":" not in path.name
        assert "BTC-USDT-USDT" in path.name
