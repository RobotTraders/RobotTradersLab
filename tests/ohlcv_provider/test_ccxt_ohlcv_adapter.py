import asyncio
import ssl
import time
from unittest.mock import AsyncMock

import aiohttp
import ccxt.async_support as ccxt
import numpy as np
import pytest

from robottraderslab import Symbol
from robottraderslab._core import (
    DownloadError,
    ExchangeConnectionError,
    InvalidTimeframeError,
    RateLimitExceededError,
)
from robottraderslab.ohlcv_provider import ccxt_ohlcv_adapter
from robottraderslab.ohlcv_provider.ccxt_ohlcv_adapter import (
    CcxtOhlcvAdapter,
)
from robottraderslab.ohlcv_provider.exceptions import InvalidSymbolError

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
JANUARY_FIRST_2024_MS = 1_704_067_200_000
BTC = Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def ccxt_exchange() -> CcxtOhlcvAdapter:
    """Create a CcxtOhlcvAdapter instance for testing."""
    return CcxtOhlcvAdapter("binance")


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())


def _adapter_against_a_stub_exchange(exchange_name: str) -> CcxtOhlcvAdapter:
    adapter = CcxtOhlcvAdapter(exchange_name)
    stub_exchange = AsyncMock()
    stub_exchange.symbols = [str(BTC)]
    stub_exchange.timeframes = {"1h": "1 hour", "1d": "1 day"}
    stub_exchange.fetch_ohlcv.return_value = []
    adapter._exchange = stub_exchange
    return adapter


class TestCcxtOhlcvAdapterInterface:
    """Test that CcxtOhlcvAdapter exposes the expected API behaviour."""

    def test_provides_expected_methods(self, ccxt_exchange: CcxtOhlcvAdapter):
        """Minimal structural check without isinstance on Protocols."""
        assert hasattr(ccxt_exchange, "_get_supported_symbols")
        assert hasattr(ccxt_exchange, "_get_supported_timeframes")
        assert hasattr(ccxt_exchange, "fetch_ohlcv")

    def test_get_supported_timeframes_returns_set(self, ccxt_exchange):
        """Test that get_supported_timeframes returns a set."""
        timeframes = ccxt_exchange._get_supported_timeframes()
        assert isinstance(timeframes, set)
        assert len(timeframes) > 0

    def test_get_supported_timeframes_contains_common_timeframes(self, ccxt_exchange):
        """Test that common timeframes are available."""
        timeframes = ccxt_exchange._get_supported_timeframes()
        common_timeframes = {"1m", "1h", "1d"}
        assert len(common_timeframes.intersection(timeframes)) > 0


class TestExchangeSpecificBehaviour:
    """Test exchange-specific configuration and behaviour."""

    @pytest.mark.parametrize(
        ("exchange_name", "expected_page_candles"),
        [
            ("binance", 1000),
            ("BINANCE", 1000),
            ("bitget", 200),
            ("bybit", 1000),
            ("okx", 100),
            ("gate", 100),
        ],
    )
    async def test_page_size_reaching_the_exchange(
        self, exchange_name, expected_page_candles
    ):
        adapter = _adapter_against_a_stub_exchange(exchange_name)

        await adapter.fetch_ohlcv(BTC, "1h", 0, 0)

        assert adapter._exchange.fetch_ohlcv.call_args.args[3] == expected_page_candles

    def test_unsupported_exchange_raises_error(self):
        """Test that unsupported exchanges raise AttributeError at initialization."""
        with pytest.raises(AttributeError):
            CcxtOhlcvAdapter("unknown_exchange")

    @pytest.mark.parametrize(
        ("exchange_name", "expected_interval_ms"),
        [
            ("bitget", 200),
            ("binance", ccxt.binance().rateLimit),
        ],
    )
    def test_request_interval_reaching_the_exchange(
        self, exchange_name, expected_interval_ms
    ):
        adapter = CcxtOhlcvAdapter(exchange_name)

        assert adapter._exchange.rateLimit == expected_interval_ms


class TestExchangeInitialization:
    """Test exchange initialization and configuration."""

    def test_exchange_stores_name_correctly(self):
        """Test that exchange correctly stores and normalizes exchange names."""
        exchange = CcxtOhlcvAdapter("BINANCE")

        assert exchange._exchange_name == "binance"


class _RecordingExchange:
    """Stand-in for a CCXT exchange that keeps the session it was constructed with."""

    cafile = None

    def __init__(self, options: dict) -> None:
        self.session = options.get("session")
        self.symbols: list[str] = []
        self.timeframes: dict[str, str] = {}

    async def __aenter__(self) -> "_RecordingExchange":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def close(self) -> None:
        return None

    async def load_markets(self) -> dict:
        return {}


class _UnreachableExchange(_RecordingExchange):
    """Stand-in for an exchange that cannot be reached once the session is open."""

    async def load_markets(self) -> dict:
        raise ccxt.RequestTimeout("bitget GET /markets")


class TestSessionHandedToCcxt:
    @pytest.fixture
    def adapter(self) -> CcxtOhlcvAdapter:
        adapter = CcxtOhlcvAdapter("binance")
        adapter._exchange_class = _RecordingExchange
        adapter._exchange = _RecordingExchange({})
        return adapter

    async def test_resolves_host_names_through_the_operating_system(self, adapter):
        async with adapter as entered:
            resolver = entered._exchange.session.connector._resolver

        assert isinstance(resolver, aiohttp.ThreadedResolver)

    async def test_verifies_certificates(self, adapter):
        async with adapter as entered:
            ssl_context = entered._exchange.session.connector._ssl

        assert isinstance(ssl_context, ssl.SSLContext)

    async def test_releases_the_session_on_exit(self, adapter):
        async with adapter as entered:
            session = entered._exchange.session

        assert session.closed

    async def test_releases_the_session_when_entry_fails(self, adapter, monkeypatch):
        built = []
        build_session = ccxt_ohlcv_adapter._build_session

        def record_session(cafile):
            session = build_session(cafile)
            built.append(session)
            return session

        monkeypatch.setattr(ccxt_ohlcv_adapter, "_build_session", record_session)
        adapter._exchange_class = _UnreachableExchange

        with pytest.raises(ccxt.RequestTimeout):
            async with adapter:
                pass

        assert built[0].closed


class TestFetchOhlcvRange:
    @pytest.fixture
    def adapter(self) -> CcxtOhlcvAdapter:
        return _adapter_against_a_stub_exchange("binance")

    async def test_with_unsupported_symbol(self, adapter):
        unsupported = Symbol.create("DOGE/USDT:USDT")

        with pytest.raises(InvalidSymbolError):
            await adapter.fetch_ohlcv(unsupported, "1h", 0, 0)

    async def test_with_unsupported_timeframe(self, adapter):
        with pytest.raises(InvalidTimeframeError):
            await adapter.fetch_ohlcv(BTC, "3m", 0, 0)

    async def test_with_a_timeframe_the_exchange_also_offers(self, adapter):
        adapter._exchange.timeframes["3m"] = "3 minutes"

        candles = await adapter.fetch_ohlcv(BTC, "3m", 0, 0)

        assert candles.size == 0

    async def test_with_a_timeframe_only_the_exchange_offers(self, adapter):
        adapter._exchange.timeframes["12h"] = "12 hours"

        with pytest.raises(InvalidTimeframeError):
            await adapter.fetch_ohlcv(BTC, "12h", 0, 0)

    async def test_with_ddos_protection(self, adapter):
        adapter._exchange.fetch_ohlcv.side_effect = ccxt.DDoSProtection("blocked")

        with pytest.raises(RateLimitExceededError):
            await adapter.fetch_ohlcv(BTC, "1h", 0, 0)

    async def test_with_rate_limit_exceeded(self, adapter):
        adapter._exchange.fetch_ohlcv.side_effect = ccxt.RateLimitExceeded("slow down")

        with pytest.raises(RateLimitExceededError):
            await adapter.fetch_ohlcv(BTC, "1h", 0, 0)

    async def test_with_network_error(self, adapter):
        adapter._exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("timeout")

        with pytest.raises(ExchangeConnectionError):
            await adapter.fetch_ohlcv(BTC, "1h", 0, 0)

    async def test_with_unexpected_exchange_error(self, adapter):
        adapter._exchange.fetch_ohlcv.side_effect = RuntimeError("unknown")

        with pytest.raises(DownloadError, match="Failed to fetch"):
            await adapter.fetch_ohlcv(BTC, "1h", 0, 0)

    async def test_range_wider_than_one_page(self, adapter):
        limit = ccxt_ohlcv_adapter.EXCHANGES["binance"]["limit_size_request"]
        page_ms = (limit - 1) * HOUR_MS

        await adapter.fetch_ohlcv(BTC, "1h", 0, 2 * page_ms)

        requested_starts = sorted(
            call.args[2] for call in adapter._exchange.fetch_ohlcv.call_args_list
        )
        assert requested_starts == [0, page_ms, 2 * page_ms]

    async def test_the_candle_in_progress(self, adapter):
        closed_at = (int(time.time() * 1000) // HOUR_MS) * HOUR_MS - HOUR_MS
        adapter._exchange.fetch_ohlcv.return_value = [
            [closed_at, 1.0, 2.0, 0.5, 1.5, 10.0],
            [closed_at + HOUR_MS, 1.0, 2.0, 0.5, 1.5, 10.0],
        ]

        candles = await adapter.fetch_ohlcv(BTC, "1h", closed_at, closed_at)

        np.testing.assert_array_equal(candles[:, 0], [closed_at])


def _two_bitget_pages(start_ms: int, period_ms: int) -> tuple[int, np.ndarray]:
    bitget = ccxt_ohlcv_adapter.EXCHANGES["bitget"]
    span_ms = bitget["request_span_days"] * DAY_MS
    limit = min(bitget["limit_size_request"], span_ms // period_ms)
    end_ms = start_ms + 2 * limit * period_ms
    return end_ms, np.arange(start_ms, end_ms + 1, period_ms)


class TestWindowsOnBitgetsTwoEndpoints:
    @pytest.mark.parametrize(
        ("timeframe", "period_ms"),
        [("1h", HOUR_MS), ("1d", DAY_MS)],
        ids=["hourly", "daily_within_the_request_span"],
    )
    async def test_a_window_the_recent_endpoint_answers(
        self, bitget_answering_from_its_recent_endpoint, timeframe, period_ms
    ):
        adapter = CcxtOhlcvAdapter("bitget")
        start_ms = JANUARY_FIRST_2024_MS
        end_ms, window_stamps = _two_bitget_pages(start_ms, period_ms)

        candles = await adapter.fetch_ohlcv(BTC, timeframe, start_ms, end_ms)

        assert np.isin(window_stamps, candles[:, 0]).all()

    @pytest.mark.parametrize(
        ("timeframe", "period_ms"),
        [("1h", HOUR_MS), ("1d", DAY_MS)],
        ids=["hourly", "daily_within_the_request_span"],
    )
    async def test_a_window_the_history_endpoint_answers(
        self, bitget_answering_from_its_history_endpoint, timeframe, period_ms
    ):
        adapter = CcxtOhlcvAdapter("bitget")
        start_ms = JANUARY_FIRST_2024_MS
        end_ms, window_stamps = _two_bitget_pages(start_ms, period_ms)

        candles = await adapter.fetch_ohlcv(BTC, timeframe, start_ms, end_ms)

        assert np.isin(window_stamps, candles[:, 0]).all()
