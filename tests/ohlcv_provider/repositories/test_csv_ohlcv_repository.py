import asyncio
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import pytz

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from robottraderslab._core import DataError
from robottraderslab.ohlcv_provider.repositories.csv_ohlcv_repository import (
    CsvOhlcvRepository,
)


@pytest.fixture
def csv_repository(temp_storage_dir: Path) -> CsvOhlcvRepository:
    return CsvOhlcvRepository("testexchange", temp_storage_dir)


class TestInitialisation:
    def test_initialises_with_exchange_name(self, temp_storage_dir):
        repo = CsvOhlcvRepository("bitget", temp_storage_dir)

        assert repo.exchange_name == "bitget"

    def test_preserves_original_exchange_name(self, temp_storage_dir):
        repo = CsvOhlcvRepository("ccxt_bitget", temp_storage_dir)

        assert repo.exchange_name == "ccxt_bitget"

    def test_creates_base_directory(self, temp_storage_dir):
        CsvOhlcvRepository("testexchange", temp_storage_dir)

        assert temp_storage_dir.exists()


class TestStore:
    @pytest.mark.asyncio
    async def test_stores_data_successfully(self, csv_repository, sample_ohlcv_data):
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        expected_path = (
            csv_repository._base_path / "testexchange" / "1h" / "BTCUSDT.csv"
        )
        assert expected_path.is_file()

    @pytest.mark.asyncio
    async def test_ccxt_and_native_share_same_folder(
        self, temp_storage_dir, sample_ohlcv_data
    ):
        repo_ccxt = CsvOhlcvRepository("ccxt_bitget", temp_storage_dir)
        repo_native = CsvOhlcvRepository("bitget", temp_storage_dir)

        await repo_ccxt.store("BTCUSDT", "1h", sample_ohlcv_data)

        loaded_from_native = await repo_native.load("BTCUSDT", "1h")
        assert not loaded_from_native.empty
        assert len(loaded_from_native) == len(sample_ohlcv_data)
        expected_path = temp_storage_dir / "bitget" / "1h" / "BTCUSDT.csv"
        assert expected_path.is_file()
        unexpected_path = temp_storage_dir / "ccxt_bitget" / "1h" / "BTCUSDT.csv"
        assert not unexpected_path.exists()

    @pytest.mark.asyncio
    async def test_empty_dataframe(self, csv_repository):
        await csv_repository.store("BTCUSDT", "1h", pd.DataFrame())

        expected_path = (
            csv_repository._base_path / "testexchange" / "1h" / "BTCUSDT.csv"
        )
        assert not expected_path.exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("symbol", "timeframe"),
        [
            ("BTCUSDT", "1h"),
            ("ETH/USDT", "1d"),
            ("BTC:USDT", "4h"),
            ("ADA-USDT", "15m"),
        ],
    )
    async def test_various_symbol_formats(
        self, csv_repository, sample_ohlcv_data, symbol, timeframe
    ):
        await csv_repository.store(symbol, timeframe, sample_ohlcv_data)

        loaded = await csv_repository.load(symbol, timeframe)
        assert not loaded.empty
        assert len(loaded) == len(sample_ohlcv_data)

    @pytest.mark.asyncio
    async def test_merges_with_existing_data(
        self, csv_repository, sample_ohlcv_data, additional_ohlcv_data
    ):
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)
        await csv_repository.store("BTCUSDT", "1h", additional_ohlcv_data)

        loaded = await csv_repository.load("BTCUSDT", "1h")

        expected_length = len(sample_ohlcv_data) + len(additional_ohlcv_data)
        assert len(loaded) == expected_length

    @pytest.mark.asyncio
    async def test_removes_duplicate_timestamps(
        self, csv_repository, sample_ohlcv_data
    ):
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        loaded = await csv_repository.load("BTCUSDT", "1h")

        assert len(loaded) == len(sample_ohlcv_data)
        assert loaded.index.is_unique


class TestLoad:
    @pytest.mark.asyncio
    async def test_round_trip_data_integrity(self, csv_repository, sample_ohlcv_data):
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        loaded = await csv_repository.load("BTCUSDT", "1h")

        loaded.index.name = None
        loaded.index.freq = None
        sample_ohlcv_data.index.name = None
        sample_ohlcv_data.index.freq = None
        pd.testing.assert_frame_equal(
            loaded.sort_index(), sample_ohlcv_data.sort_index()
        )

    @pytest.mark.asyncio
    async def test_loaded_index_is_utc_and_sorted(
        self, csv_repository, sample_ohlcv_data
    ):
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        loaded = await csv_repository.load("BTCUSDT", "1h")

        assert str(loaded.index.tz).upper() == "UTC"
        assert loaded.index.is_monotonic_increasing

    @pytest.mark.asyncio
    async def test_nonexistent_data(self, csv_repository):
        loaded = await csv_repository.load("NONEXISTENT", "1h")

        assert loaded.empty

    @pytest.mark.asyncio
    async def test_filters_by_date_range(self, csv_repository, sample_ohlcv_data):
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)
        start = datetime(2024, 1, 1, 2, 0, 0, tzinfo=pytz.UTC)
        end = datetime(2024, 1, 1, 4, 0, 0, tzinfo=pytz.UTC)

        filtered = await csv_repository.load("BTCUSDT", "1h", start, end)

        assert 0 < len(filtered) < len(sample_ohlcv_data)

    @pytest.mark.asyncio
    async def test_start_date_only(self, csv_repository, sample_ohlcv_data):
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)
        start = datetime(2024, 1, 1, 2, 0, 0, tzinfo=pytz.UTC)

        filtered = await csv_repository.load("BTCUSDT", "1h", start_date=start)

        assert len(filtered) < len(sample_ohlcv_data)

    @pytest.mark.asyncio
    async def test_end_date_only(self, csv_repository, sample_ohlcv_data):
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)
        end = datetime(2024, 1, 1, 2, 0, 0, tzinfo=pytz.UTC)

        filtered = await csv_repository.load("BTCUSDT", "1h", end_date=end)

        assert len(filtered) < len(sample_ohlcv_data)


class TestExists:
    @pytest.mark.asyncio
    async def test_stored_data(self, csv_repository, sample_ohlcv_data):
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        assert await csv_repository.exists("BTCUSDT", "1h")

    @pytest.mark.asyncio
    async def test_nonexistent_data(self, csv_repository):
        assert not await csv_repository.exists("NONEXISTENT", "1h")

    @pytest.mark.asyncio
    async def test_date_range_coverage(self, csv_repository, sample_ohlcv_data):
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        covered_start = datetime(2024, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        covered_end = datetime(2024, 1, 1, 3, 0, 0, tzinfo=pytz.UTC)
        assert await csv_repository.exists("BTCUSDT", "1h", covered_start, covered_end)

        outside_start = datetime(2023, 12, 31, 0, 0, 0, tzinfo=pytz.UTC)
        outside_end = datetime(2023, 12, 31, 23, 0, 0, tzinfo=pytz.UTC)
        assert not await csv_repository.exists(
            "BTCUSDT", "1h", outside_start, outside_end
        )

    @pytest.mark.asyncio
    async def test_with_end_date_beyond_data_range(
        self, csv_repository, sample_ohlcv_data
    ):
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        covered_start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
        beyond_end = datetime(2025, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)

        assert not await csv_repository.exists(
            "BTCUSDT", "1h", covered_start, beyond_end
        )


class TestDelete:
    @pytest.mark.asyncio
    async def test_deletes_existing_data(self, csv_repository, sample_ohlcv_data):
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)
        assert await csv_repository.exists("BTCUSDT", "1h")

        await csv_repository.delete("BTCUSDT", "1h")

        assert not await csv_repository.exists("BTCUSDT", "1h")

    @pytest.mark.asyncio
    async def test_nonexistent_data(self, csv_repository):
        await csv_repository.delete("NONEXISTENT", "1h")


class TestAtomicWrite:
    @pytest.mark.asyncio
    async def test_existing_data_preserved_on_write_failure(
        self, csv_repository, sample_ohlcv_data
    ):
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        with patch.object(
            csv_repository, "_write_file", side_effect=OSError("disk full")
        ):
            with pytest.raises(DataError):
                await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        preserved = await csv_repository.load("BTCUSDT", "1h")

        assert len(preserved) == len(sample_ohlcv_data)

    @pytest.mark.asyncio
    async def test_no_temp_files_on_write_failure(
        self, csv_repository, sample_ohlcv_data
    ):
        with patch.object(
            csv_repository, "_write_file", side_effect=OSError("disk full")
        ):
            with pytest.raises(DataError):
                await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        file_path = csv_repository._build_file_path("BTCUSDT", "1h")
        leftover_tmp_files = list(file_path.parent.glob("*.tmp"))

        assert leftover_tmp_files == []


class TestCorruptedCacheRecovery:
    @pytest.mark.asyncio
    async def test_empty_file_is_discarded(self, csv_repository):
        file_path = csv_repository._build_file_path("BTCUSDT", "1h")
        file_path.touch()

        loaded = await csv_repository.load("BTCUSDT", "1h")

        assert loaded.empty
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_empty_file_logs_warning(self, csv_repository, caplog):
        file_path = csv_repository._build_file_path("BTCUSDT", "1h")
        file_path.touch()

        with caplog.at_level("WARNING"):
            await csv_repository.load("BTCUSDT", "1h")

        assert "Corrupted cache file (empty)" in caplog.text

    @pytest.mark.asyncio
    async def test_partially_written_file_is_discarded(self, csv_repository):
        file_path = csv_repository._build_file_path("BTCUSDT", "1h")
        file_path.write_text("date,open,high,low,close,volume\n2024-01-01,,,,,\n")

        loaded = await csv_repository.load("BTCUSDT", "1h")

        assert loaded.empty
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_partially_written_file_logs_warning(self, csv_repository, caplog):
        file_path = csv_repository._build_file_path("BTCUSDT", "1h")
        file_path.write_text("date,open,high,low,close,volume\n2024-01-01,,,,,\n")

        with caplog.at_level("WARNING"):
            await csv_repository.load("BTCUSDT", "1h")

        assert "Corrupted cache file (unparseable)" in caplog.text

    @pytest.mark.asyncio
    async def test_valid_file_is_not_affected(self, csv_repository, sample_ohlcv_data):
        await csv_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        loaded = await csv_repository.load("BTCUSDT", "1h")

        assert not loaded.empty
        assert len(loaded) == len(sample_ohlcv_data)
