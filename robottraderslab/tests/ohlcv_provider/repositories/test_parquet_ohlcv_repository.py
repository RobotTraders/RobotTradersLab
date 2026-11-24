import asyncio
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import pytz

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from robottraderslab.ohlcv_provider.repositories.parquet_ohlcv_repository import (
    ParquetOhlcvRepository,
)


@pytest.fixture
def parquet_repository(temp_storage_dir: Path) -> ParquetOhlcvRepository:
    return ParquetOhlcvRepository("testexchange", temp_storage_dir)


class TestInitialisation:
    def test_initialises_with_exchange_name(self, temp_storage_dir):
        repo = ParquetOhlcvRepository("bitget", temp_storage_dir)

        assert repo.exchange_name == "bitget"

    def test_preserves_original_exchange_name(self, temp_storage_dir):
        repo = ParquetOhlcvRepository("ccxt_bitget", temp_storage_dir)

        assert repo.exchange_name == "ccxt_bitget"

    def test_creates_base_directory(self, temp_storage_dir):
        ParquetOhlcvRepository("testexchange", temp_storage_dir)

        assert temp_storage_dir.exists()


class TestStore:
    @pytest.mark.asyncio
    async def test_stores_data_successfully(
        self, parquet_repository, sample_ohlcv_data
    ):
        await parquet_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        expected_path = (
            parquet_repository._base_path / "testexchange" / "1h" / "BTCUSDT.parquet"
        )
        assert expected_path.is_file()

    @pytest.mark.asyncio
    async def test_ccxt_and_native_share_same_folder(
        self, temp_storage_dir, sample_ohlcv_data
    ):
        repo_ccxt = ParquetOhlcvRepository("ccxt_bitget", temp_storage_dir)
        repo_native = ParquetOhlcvRepository("bitget", temp_storage_dir)

        await repo_ccxt.store("BTCUSDT", "1h", sample_ohlcv_data)

        loaded_from_native = await repo_native.load("BTCUSDT", "1h")
        assert not loaded_from_native.empty
        assert len(loaded_from_native) == len(sample_ohlcv_data)
        expected_path = temp_storage_dir / "bitget" / "1h" / "BTCUSDT.parquet"
        assert expected_path.is_file()
        unexpected_path = temp_storage_dir / "ccxt_bitget" / "1h" / "BTCUSDT.parquet"
        assert not unexpected_path.exists()

    @pytest.mark.asyncio
    async def test_empty_dataframe(self, parquet_repository):
        await parquet_repository.store("BTCUSDT", "1h", pd.DataFrame())

        expected_path = (
            parquet_repository._base_path / "testexchange" / "1h" / "BTCUSDT.parquet"
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
        self, parquet_repository, sample_ohlcv_data, symbol, timeframe
    ):
        await parquet_repository.store(symbol, timeframe, sample_ohlcv_data)

        loaded = await parquet_repository.load(symbol, timeframe)
        assert not loaded.empty
        assert len(loaded) == len(sample_ohlcv_data)

    @pytest.mark.asyncio
    async def test_merges_with_existing_data(
        self, parquet_repository, sample_ohlcv_data, additional_ohlcv_data
    ):
        await parquet_repository.store("BTCUSDT", "1h", sample_ohlcv_data)
        await parquet_repository.store("BTCUSDT", "1h", additional_ohlcv_data)

        loaded = await parquet_repository.load("BTCUSDT", "1h")

        expected_length = len(sample_ohlcv_data) + len(additional_ohlcv_data)
        assert len(loaded) == expected_length

    @pytest.mark.asyncio
    async def test_removes_duplicate_timestamps(
        self, parquet_repository, sample_ohlcv_data
    ):
        await parquet_repository.store("BTCUSDT", "1h", sample_ohlcv_data)
        await parquet_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        loaded = await parquet_repository.load("BTCUSDT", "1h")

        assert len(loaded) == len(sample_ohlcv_data)
        assert loaded.index.is_unique


class TestLoad:
    @pytest.mark.asyncio
    async def test_round_trip_data_integrity(
        self, parquet_repository, sample_ohlcv_data
    ):
        await parquet_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        loaded = await parquet_repository.load("BTCUSDT", "1h")

        loaded.index.name = None
        loaded.index.freq = None
        sample_ohlcv_data.index.name = None
        sample_ohlcv_data.index.freq = None
        pd.testing.assert_frame_equal(
            loaded.sort_index(), sample_ohlcv_data.sort_index()
        )

    @pytest.mark.asyncio
    async def test_loaded_index_is_utc_and_sorted(
        self, parquet_repository, sample_ohlcv_data
    ):
        await parquet_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        loaded = await parquet_repository.load("BTCUSDT", "1h")

        assert str(loaded.index.tz).upper() == "UTC"
        assert loaded.index.is_monotonic_increasing

    @pytest.mark.asyncio
    async def test_nonexistent_data(self, parquet_repository):
        loaded = await parquet_repository.load("NONEXISTENT", "1h")

        assert loaded.empty

    @pytest.mark.asyncio
    async def test_filters_by_date_range(self, parquet_repository, sample_ohlcv_data):
        await parquet_repository.store("BTCUSDT", "1h", sample_ohlcv_data)
        start = datetime(2024, 1, 1, 2, 0, 0, tzinfo=pytz.UTC)
        end = datetime(2024, 1, 1, 4, 0, 0, tzinfo=pytz.UTC)

        filtered = await parquet_repository.load("BTCUSDT", "1h", start, end)

        assert 0 < len(filtered) < len(sample_ohlcv_data)

    @pytest.mark.asyncio
    async def test_start_date_only(self, parquet_repository, sample_ohlcv_data):
        await parquet_repository.store("BTCUSDT", "1h", sample_ohlcv_data)
        start = datetime(2024, 1, 1, 2, 0, 0, tzinfo=pytz.UTC)

        filtered = await parquet_repository.load("BTCUSDT", "1h", start_date=start)

        assert len(filtered) < len(sample_ohlcv_data)

    @pytest.mark.asyncio
    async def test_end_date_only(self, parquet_repository, sample_ohlcv_data):
        await parquet_repository.store("BTCUSDT", "1h", sample_ohlcv_data)
        end = datetime(2024, 1, 1, 2, 0, 0, tzinfo=pytz.UTC)

        filtered = await parquet_repository.load("BTCUSDT", "1h", end_date=end)

        assert len(filtered) < len(sample_ohlcv_data)


class TestExists:
    @pytest.mark.asyncio
    async def test_stored_data(self, parquet_repository, sample_ohlcv_data):
        await parquet_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        assert await parquet_repository.exists("BTCUSDT", "1h")

    @pytest.mark.asyncio
    async def test_nonexistent_data(self, parquet_repository):
        assert not await parquet_repository.exists("NONEXISTENT", "1h")

    @pytest.mark.asyncio
    async def test_date_range_coverage(self, parquet_repository, sample_ohlcv_data):
        await parquet_repository.store("BTCUSDT", "1h", sample_ohlcv_data)

        covered_start = datetime(2024, 1, 1, 1, 0, 0, tzinfo=pytz.UTC)
        covered_end = datetime(2024, 1, 1, 3, 0, 0, tzinfo=pytz.UTC)
        assert await parquet_repository.exists(
            "BTCUSDT", "1h", covered_start, covered_end
        )

        outside_start = datetime(2023, 12, 31, 0, 0, 0, tzinfo=pytz.UTC)
        outside_end = datetime(2023, 12, 31, 23, 0, 0, tzinfo=pytz.UTC)
        assert not await parquet_repository.exists(
            "BTCUSDT", "1h", outside_start, outside_end
        )


class TestDelete:
    @pytest.mark.asyncio
    async def test_deletes_existing_data(self, parquet_repository, sample_ohlcv_data):
        await parquet_repository.store("BTCUSDT", "1h", sample_ohlcv_data)
        assert await parquet_repository.exists("BTCUSDT", "1h")

        await parquet_repository.delete("BTCUSDT", "1h")

        assert not await parquet_repository.exists("BTCUSDT", "1h")

    @pytest.mark.asyncio
    async def test_nonexistent_data(self, parquet_repository):
        await parquet_repository.delete("NONEXISTENT", "1h")
