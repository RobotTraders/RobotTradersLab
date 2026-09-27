import asyncio
import sys
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Self
from unittest.mock import Mock, patch

import pandas as pd
import pytest

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from robottraderslab import Symbol, TimeFrame
from robottraderslab.exchanges import OhlcvAdapterProtocol
from robottraderslab.ohlcv_provider.exceptions import StorageTypeError
from robottraderslab.ohlcv_provider.exchange.ohlcv_adapter_factory import (
    OhlcvAdapterFactory,
)
from robottraderslab.ohlcv_provider.exchange.ohlcv_provider import (
    ExchangeOHLCVProvider,
    _create_repository,
)


class MockExchangeAdapter(OhlcvAdapterProtocol):
    """Test exchange adapter that simulates real exchange behaviour with controlled test data."""

    def __init__(self, exchange_name: str):
        self._exchange_name = exchange_name
        self._test_data = {
            ("BTC/USDT:USDT", "1h"): [
                [
                    1704067200000,
                    100.0,
                    105.0,
                    99.0,
                    101.0,
                    1000.0,
                ],  # 2024-01-01 00:00:00
                [
                    1704070800000,
                    101.0,
                    106.0,
                    100.0,
                    102.0,
                    1100.0,
                ],  # 2024-01-01 01:00:00
                [
                    1704074400000,
                    102.0,
                    107.0,
                    101.0,
                    103.0,
                    1200.0,
                ],  # 2024-01-01 02:00:00
            ],
            ("ETH/USDT:USDT", "1h"): [
                [1704067200000, 2000.0, 2050.0, 1990.0, 2010.0, 500.0],
                [1704070800000, 2010.0, 2060.0, 2000.0, 2020.0, 550.0],
            ],
        }

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    def _get_supported_symbols(self) -> set[Symbol]:
        return {
            Symbol.create("BTC/USDT:USDT"),
            Symbol.create("ETH/USDT:USDT"),
        }

    def _get_supported_timeframes(self) -> set[str]:
        return {"1h", "4h", "1d"}

    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_ms: int,
        end_ms: int,
    ) -> list[list[float]]:
        """Return test data for the requested symbol/timeframe."""
        key = (str(symbol), str(timeframe))
        if key not in self._test_data:
            return []

        return [row for row in self._test_data[key] if start_ms <= row[0] <= end_ms]


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    with TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def test_symbol() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


async def _open_adapter(factory: OhlcvAdapterFactory, adapter: str) -> None:
    async with factory.using(adapter):
        pass


class TestExchangeOHLCVProviderIntegration:
    """
    Integration tests for ExchangeOHLCVProvider orchestration logic.

    These tests focus ONLY on how ExchangeOHLCVProvider integrates its components:
    - Date parsing and validation
    - Factory usage (_create_repository, OhlcvAdapterFactory in service layer)
    - Service orchestration (calling fetch_ohlcv_with with correct parameters)
    - Async/sync boundary handling (asyncio.run)

    Note: Individual components (repository, service, exchange) are tested separately.
    """

    def test_provider_parses_dates_and_calls_service_correctly(
        self, temp_dir, test_symbol, stub_exchange
    ):
        with (
            patch(
                "robottraderslab.ohlcv_provider.exchange.ohlcv_provider.fetch_ohlcv_with"
            ) as mock_service,
            patch(
                "robottraderslab.ohlcv_provider.exchange.ohlcv_provider.get_cached_ohlcv"
            ) as mock_get_cache,
            patch(
                "robottraderslab.ohlcv_provider.exchange.ohlcv_provider.store_ohlcv_in_cache"
            ),
        ):
            mock_service.return_value = pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [105.0],
                    "low": [99.0],
                    "close": [101.0],
                    "volume": [1000.0],
                }
            )
            mock_get_cache.return_value = None

            provider = ExchangeOHLCVProvider(
                exchange=stub_exchange,
                storage_dir=temp_dir,
            )
            provider.set_dates("2024-01-01", "2024-01-02")

            provider.fetch_ohlcv(test_symbol, "1h")

            mock_service.assert_called_once()
            call_args = mock_service.call_args

            start_date_arg = call_args.kwargs["start_date"]
            end_date_arg = call_args.kwargs["end_date"]

            assert start_date_arg.year == 2024
            assert start_date_arg.month == 1
            assert start_date_arg.day == 1
            assert end_date_arg.day == 2

    def test_provider_uses_factory_functions_correctly(
        self, temp_dir, test_symbol, stub_exchange
    ):
        with (
            patch(
                "robottraderslab.ohlcv_provider.exchange.ohlcv_provider._create_repository"
            ) as mock_repo_factory,
            patch(
                "robottraderslab.ohlcv_provider.exchange.ohlcv_provider.fetch_ohlcv_with"
            ) as mock_service,
            patch(
                "robottraderslab.ohlcv_provider.exchange.ohlcv_provider.get_cached_ohlcv"
            ) as mock_get_cache,
            patch(
                "robottraderslab.ohlcv_provider.exchange.ohlcv_provider.store_ohlcv_in_cache"
            ),
        ):
            mock_repo = Mock()
            mock_repo_factory.return_value = mock_repo
            mock_service.return_value = pd.DataFrame()
            mock_get_cache.return_value = None

            provider = ExchangeOHLCVProvider(
                exchange=stub_exchange,
                storage_dir=temp_dir,
            )
            provider.set_dates("2024-01-01", "2024-01-02")

            provider.fetch_ohlcv(test_symbol, "1h")

            mock_repo_factory.assert_called_once_with(
                "parquet", stub_exchange, dir=temp_dir
            )

            mock_service.assert_called_once()
            call_args = mock_service.call_args
            assert call_args.kwargs["repository"] is mock_repo
            assert call_args.kwargs["dataset_name"] == stub_exchange

    def test_provider_handles_async_boundary_correctly(
        self, temp_dir, test_symbol, stub_exchange
    ):
        fetched = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [9.0],
            }
        )

        async def fetch(**_kwargs):
            return fetched

        with (
            patch(
                "robottraderslab.ohlcv_provider.exchange.ohlcv_provider.fetch_ohlcv_with",
                side_effect=fetch,
            ),
            patch(
                "robottraderslab.ohlcv_provider.exchange.ohlcv_provider.get_cached_ohlcv"
            ) as mock_get_cache,
            patch(
                "robottraderslab.ohlcv_provider.exchange.ohlcv_provider.store_ohlcv_in_cache"
            ) as mock_store_cache,
        ):
            mock_get_cache.return_value = None

            provider = ExchangeOHLCVProvider(
                exchange=stub_exchange,
                storage_dir=temp_dir,
            )
            provider.set_dates("2024-01-01", "2024-01-02")

            candles = provider.fetch_ohlcv(test_symbol, "1h")

            assert candles.equals(fetched)
            mock_get_cache.assert_called_once()
            mock_store_cache.assert_called_once()

    def test_provider_factory_functions_work_independently(self, temp_dir):
        """Test the factory functions work correctly when called directly."""
        repo = _create_repository("csv", "test_exchange", dir=temp_dir)
        assert hasattr(repo, "exchange_name")
        assert repo.exchange_name == "test_exchange"

        with pytest.raises(StorageTypeError, match="Unsupported storage_type"):
            _create_repository("unsupported", "test", dir=temp_dir)

        factory = OhlcvAdapterFactory()
        with pytest.raises(RuntimeError, match="OHLCV adapter .* not found"):
            asyncio.run(_open_adapter(factory, "unsupported_nonexistent_test"))
