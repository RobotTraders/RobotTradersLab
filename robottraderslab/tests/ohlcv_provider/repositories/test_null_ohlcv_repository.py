import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.ohlcv_provider.repositories.null_ohlcv_repository import (
    NullOhlcvRepository,
)


@pytest.fixture
def repository() -> NullOhlcvRepository:
    return NullOhlcvRepository()


@pytest.fixture
def symbol() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


class TestNullOhlcvRepository:
    @pytest.mark.asyncio
    async def test_load_without_stored_data(self, repository, symbol):
        loaded_data = await repository.load(symbol, "1h")

        assert loaded_data.empty

    @pytest.mark.asyncio
    async def test_store_does_not_persist(self, repository, symbol):
        ohlcv_data = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100.0],
            }
        )

        await repository.store(symbol, "1h", ohlcv_data)

        loaded_data = await repository.load(symbol, "1h")
        assert loaded_data.empty

    @pytest.mark.asyncio
    async def test_exists_without_stored_data(self, repository, symbol):
        has_data = await repository.exists(symbol, "1h")

        assert has_data is False

    @pytest.mark.asyncio
    async def test_delete_without_stored_data(self, repository, symbol):
        await repository.delete(symbol, "1h")
