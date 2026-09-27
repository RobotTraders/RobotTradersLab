from unittest.mock import Mock

import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab._core import OHLCVProviderProtocol
from robottraderslab.analyser.reference_price_handler import (
    resolve_reference,
    resolve_reference_price,
)
from robottraderslab.exceptions import ExchangeRecoverableError

_SYMBOL = Symbol.create("BTC/USDT:USDT")
_VENUE_CLOSES = [105.0, 107.0, 109.0, 111.0, 113.0]
_CACHED_CLOSES = [200.0, 201.0, 202.0, 203.0, 204.0]


@pytest.fixture
def daily_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=5, freq="D")


@pytest.fixture
def provider(daily_index) -> Mock:
    """A venue whose cache holds nothing for the symbol."""
    provider = Mock(spec=OHLCVProviderProtocol)
    provider.get_all_cached_ohlcv_for_symbol.return_value = pd.DataFrame()
    provider.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": _VENUE_CLOSES,
            "volume": 1000.0,
        },
        index=daily_index,
    )
    return provider


class TestResolveReferencePrice:
    def test_cached_candles_are_read_before_the_venue(self, provider, daily_index):
        provider.get_all_cached_ohlcv_for_symbol.return_value = pd.DataFrame(
            {"close": _CACHED_CLOSES}, index=daily_index
        )

        reference_price = resolve_reference_price(
            _SYMBOL, daily_index, provider, preferred_timeframe=None
        )

        assert reference_price.tolist() == _CACHED_CLOSES
        provider.fetch_ohlcv.assert_not_called()

    def test_an_empty_cache_is_filled_from_the_venue(self, provider, daily_index):
        reference_price = resolve_reference_price(
            _SYMBOL, daily_index, provider, preferred_timeframe=None
        )

        assert reference_price.tolist() == _VENUE_CLOSES

    def test_daily_equity_reads_daily_candles(self, provider, daily_index):
        resolve_reference_price(
            _SYMBOL, daily_index, provider, preferred_timeframe=None
        )

        provider.fetch_ohlcv.assert_called_once_with(_SYMBOL, "1d")

    def test_hourly_equity_reads_hourly_candles(self, provider):
        hourly_index = pd.date_range("2024-01-01", periods=5, freq="h")

        resolve_reference_price(
            _SYMBOL, hourly_index, provider, preferred_timeframe=None
        )

        provider.fetch_ohlcv.assert_called_once_with(_SYMBOL, "1h")

    def test_45_minute_equity_reads_45_minute_candles(self, provider):
        forty_five_minute_index = pd.date_range("2024-01-01", periods=5, freq="45min")

        resolve_reference_price(
            _SYMBOL, forty_five_minute_index, provider, preferred_timeframe=None
        )

        provider.fetch_ohlcv.assert_called_once_with(_SYMBOL, "45m")

    def test_sub_minute_equity_reads_the_smallest_candles(self, provider):
        sub_minute_index = pd.date_range("2024-01-01", periods=5, freq="30s")

        resolve_reference_price(
            _SYMBOL, sub_minute_index, provider, preferred_timeframe=None
        )

        provider.fetch_ohlcv.assert_called_once_with(_SYMBOL, "1m")

    def test_12_hour_equity_reads_4_hour_candles(self, provider):
        twelve_hour_index = pd.date_range("2024-01-01", periods=5, freq="12h")

        resolve_reference_price(
            _SYMBOL, twelve_hour_index, provider, preferred_timeframe=None
        )

        provider.fetch_ohlcv.assert_called_once_with(_SYMBOL, "4h")

    def test_a_preferred_timeframe_overrides_the_equity_spacing(
        self, provider, daily_index
    ):
        resolve_reference_price(
            _SYMBOL, daily_index, provider, preferred_timeframe="4h"
        )

        provider.fetch_ohlcv.assert_called_once_with(_SYMBOL, "4h")

    def test_a_venue_failure_leaves_the_run_uncompared(self, provider, daily_index):
        provider.fetch_ohlcv.side_effect = ExchangeRecoverableError("API Error")

        assert (
            resolve_reference_price(
                _SYMBOL, daily_index, provider, preferred_timeframe=None
            )
            is None
        )


class TestResolveReference:
    def test_the_declared_text_reaches_the_provider_parsed(self, provider, daily_index):
        resolve_reference("BTC/USDT:USDT", None, daily_index, provider)

        asked_for = provider.fetch_ohlcv.call_args.args[0]
        assert isinstance(asked_for, Symbol)
        assert asked_for.base == "BTC"

    def test_the_reference_is_labelled_by_the_symbol_it_priced(
        self, provider, daily_index
    ):
        _, label = resolve_reference("BTC/USDT:USDT", None, daily_index, provider)

        assert label == "BTC/USDT:USDT"

    def test_a_declared_timeframe_is_the_one_fetched(self, provider, daily_index):
        resolve_reference("BTC/USDT:USDT", "4h", daily_index, provider)

        provider.fetch_ohlcv.assert_called_once_with(_SYMBOL, "4h")

    def test_a_run_declaring_no_reference_asks_the_venue_for_nothing(
        self, provider, daily_index
    ):
        price, label = resolve_reference(None, None, daily_index, provider)

        assert price is None
        assert label is None
        provider.fetch_ohlcv.assert_not_called()
