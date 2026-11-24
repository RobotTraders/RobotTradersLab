import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.ohlcv_provider.exchange.ohlcv_cache import (
    _OHLCV_CACHE,
    get_all_cached_ohlcv_for_symbol,
    get_cached_ohlcv,
    store_ohlcv_in_cache,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    )


@pytest.fixture
def btc_symbol() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def eth_symbol() -> Symbol:
    return Symbol.create("ETH/USDT:USDT")


@pytest.fixture
def _clean_cache():
    _OHLCV_CACHE.clear()
    yield
    _OHLCV_CACHE.clear()


@pytest.mark.usefixtures("_clean_cache")
class TestGetCachedOhlcv:
    def test_returns_none_when_cache_empty(self, btc_symbol):
        cached_ohlcv = get_cached_ohlcv(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02"
        )
        assert cached_ohlcv is None

    def test_returns_none_when_exchange_not_cached(self, btc_symbol, sample_dataframe):
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02", sample_dataframe
        )

        cached_ohlcv = get_cached_ohlcv(
            "binance", btc_symbol, "1h", "2024-01-01", "2024-01-02"
        )
        assert cached_ohlcv is None

    def test_returns_none_when_timeframe_not_cached(self, btc_symbol, sample_dataframe):
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "4h", "2024-01-01", "2024-01-02", sample_dataframe
        )

        cached_ohlcv = get_cached_ohlcv(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02"
        )
        assert cached_ohlcv is None

    def test_returns_none_when_symbol_not_cached(
        self, btc_symbol, eth_symbol, sample_dataframe
    ):
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02", sample_dataframe
        )

        cached_ohlcv = get_cached_ohlcv(
            "bitget", eth_symbol, "1h", "2024-01-01", "2024-01-02"
        )
        assert cached_ohlcv is None

    def test_returns_none_when_date_range_different(self, btc_symbol, sample_dataframe):
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02", sample_dataframe
        )

        cached_ohlcv = get_cached_ohlcv(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-03"
        )
        assert cached_ohlcv is None

    def test_returns_dataframe_when_exact_match(self, btc_symbol, sample_dataframe):
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02", sample_dataframe
        )

        cached_ohlcv = get_cached_ohlcv(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02"
        )

        assert isinstance(cached_ohlcv, pd.DataFrame)
        pd.testing.assert_frame_equal(cached_ohlcv, sample_dataframe)


@pytest.mark.usefixtures("_clean_cache")
class TestStoreOhlcvInCache:
    def test_stores_data_in_empty_cache(self, btc_symbol, sample_dataframe):
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02", sample_dataframe
        )

        assert "bitget" in _OHLCV_CACHE
        assert "1h" in _OHLCV_CACHE["bitget"]
        assert "BTC/USDT:USDT" in _OHLCV_CACHE["bitget"]["1h"]

        cached_data = _OHLCV_CACHE["bitget"]["1h"]["BTC/USDT:USDT"]
        start_date, end_date, dataframe = cached_data

        assert start_date == "2024-01-01"
        assert end_date == "2024-01-02"
        pd.testing.assert_frame_equal(dataframe, sample_dataframe)

    def test_stores_multiple_symbols_same_exchange_timeframe(
        self, btc_symbol, eth_symbol, sample_dataframe
    ):
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02", sample_dataframe
        )
        store_ohlcv_in_cache(
            "bitget", eth_symbol, "1h", "2024-01-01", "2024-01-02", sample_dataframe
        )

        assert len(_OHLCV_CACHE["bitget"]["1h"]) == 2
        assert "BTC/USDT:USDT" in _OHLCV_CACHE["bitget"]["1h"]
        assert "ETH/USDT:USDT" in _OHLCV_CACHE["bitget"]["1h"]

    def test_stores_multiple_timeframes_same_exchange_symbol(
        self, btc_symbol, sample_dataframe
    ):
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02", sample_dataframe
        )
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "4h", "2024-01-01", "2024-01-02", sample_dataframe
        )

        assert len(_OHLCV_CACHE["bitget"]) == 2
        assert "1h" in _OHLCV_CACHE["bitget"]
        assert "4h" in _OHLCV_CACHE["bitget"]

    def test_stores_multiple_exchanges(self, btc_symbol, sample_dataframe):
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02", sample_dataframe
        )
        store_ohlcv_in_cache(
            "binance", btc_symbol, "1h", "2024-01-01", "2024-01-02", sample_dataframe
        )

        assert len(_OHLCV_CACHE) == 2
        assert "bitget" in _OHLCV_CACHE
        assert "binance" in _OHLCV_CACHE

    def test_overwrites_existing_data(self, btc_symbol, sample_dataframe):
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02", sample_dataframe
        )

        new_dataframe = pd.DataFrame(
            {
                "open": [200.0],
                "high": [205.0],
                "low": [199.0],
                "close": [204.0],
                "volume": [2000.0],
            }
        )

        store_ohlcv_in_cache(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02", new_dataframe
        )

        cached_data = _OHLCV_CACHE["bitget"]["1h"]["BTC/USDT:USDT"]
        _, _, dataframe = cached_data
        pd.testing.assert_frame_equal(dataframe, new_dataframe)


@pytest.mark.usefixtures("_clean_cache")
class TestCacheIntegration:
    @pytest.mark.parametrize(
        ("exchange", "timeframe", "start_date", "end_date"),
        [
            ("bitget", "1h", "2024-01-01", "2024-01-02"),
            ("bitget", "4h", "2023-12-01", "2023-12-31"),
            ("coinbase", "1d", "2024-06-01", "2024-06-30"),
        ],
    )
    def test_store_and_retrieve_cycle(
        self,
        btc_symbol,
        sample_dataframe,
        exchange,
        timeframe,
        start_date,
        end_date,
    ):
        store_ohlcv_in_cache(
            exchange, btc_symbol, timeframe, start_date, end_date, sample_dataframe
        )

        cached_ohlcv = get_cached_ohlcv(
            exchange, btc_symbol, timeframe, start_date, end_date
        )

        assert cached_ohlcv is not None
        pd.testing.assert_frame_equal(cached_ohlcv, sample_dataframe)

    def test_cache_isolation_between_parameters(
        self, btc_symbol, eth_symbol, sample_dataframe
    ):
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02", sample_dataframe
        )
        store_ohlcv_in_cache(
            "bitget", eth_symbol, "1h", "2024-01-01", "2024-01-02", sample_dataframe
        )
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "4h", "2024-01-01", "2024-01-02", sample_dataframe
        )
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02", sample_dataframe
        )

        assert (
            get_cached_ohlcv("bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02")
            is not None
        )
        assert (
            get_cached_ohlcv("bitget", eth_symbol, "1h", "2024-01-01", "2024-01-02")
            is not None
        )
        assert (
            get_cached_ohlcv("bitget", btc_symbol, "4h", "2024-01-01", "2024-01-02")
            is not None
        )
        assert (
            get_cached_ohlcv("bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02")
            is not None
        )

        assert (
            get_cached_ohlcv("coinbase", btc_symbol, "1h", "2024-01-01", "2024-01-02")
            is None
        )
        assert (
            get_cached_ohlcv("bitget", btc_symbol, "1h", "2024-01-01", "2024-01-03")
            is None
        )


@pytest.mark.usefixtures("_clean_cache")
class TestGetAllCachedOhlcvForSymbol:
    def test_merges_across_timeframes(self, btc_symbol):
        dates_1h = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        df_1h = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [99.0, 100.0, 101.0],
                "close": [104.0, 105.0, 106.0],
                "volume": [1000.0, 1100.0, 1200.0],
            },
            index=dates_1h,
        )
        dates_1d = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
        df_1d = pd.DataFrame(
            {
                "open": [100.0, 110.0],
                "high": [115.0, 116.0],
                "low": [98.0, 109.0],
                "close": [110.0, 115.0],
                "volume": [5000.0, 5100.0],
            },
            index=dates_1d,
        )

        store_ohlcv_in_cache(
            "bitget", btc_symbol, "1h", "2024-01-01", "2024-01-02", df_1h
        )
        store_ohlcv_in_cache(
            "bitget", btc_symbol, "1d", "2024-01-01", "2024-01-02", df_1d
        )

        merged = get_all_cached_ohlcv_for_symbol("bitget", btc_symbol)

        assert len(merged) == 4
        assert merged.index.is_monotonic_increasing
        assert not merged.index.has_duplicates
