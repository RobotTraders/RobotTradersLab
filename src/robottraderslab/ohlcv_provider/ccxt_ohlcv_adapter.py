import ssl
from functools import partial
from types import TracebackType
from typing import Any, Self

import aiohttp
import ccxt.async_support as ccxt
import numpy as np

from robottraderslab._core import (
    TIMEFRAMES,
    DownloadError,
    ExchangeConnectionError,
    InvalidTimeframeError,
    RateLimitExceededError,
    Symbol,
    TimeFrame,
    to_milliseconds,
)
from robottraderslab.exchanges import (
    OhlcvAdapterProtocol,
    OhlcvData,
    fetch_pages,
    no_candles,
    page_bounds,
    without_the_open_candle,
)

from .exceptions import InvalidSymbolError

type CcxtAdapterConfig = dict[str, dict[str, Any]]

EXCHANGES: CcxtAdapterConfig = {
    "bitget": {
        "limit_size_request": 200,
        "request_interval_ms": 200,
        "request_span_days": 90,
    },
    "binance": {
        "limit_size_request": 1000,
    },
    "binanceusdm": {
        "limit_size_request": 1000,
    },
    "kucoin": {
        "limit_size_request": 200,
    },
    "kucoinfutures": {
        "limit_size_request": 200,
    },
    "okx": {
        "limit_size_request": 100,
    },
    "bybit": {
        "limit_size_request": 1000,
    },
}

DEFAULT_EXCHANGE_CONFIG = {
    "limit_size_request": 100,
}

_DAY_MS = 24 * 60 * 60 * 1000


class CcxtOhlcvAdapter(OhlcvAdapterProtocol):
    """CCXT-based adapter for OHLCV data operations."""

    def __init__(self, exchange_name: str):
        """Initialise the OHLCV exchange with CCXT.

        Args:
            exchange_name: Name of the exchange (e.g., 'binance', 'bitget')
        """
        self._exchange_name = exchange_name.lower()
        self._exchange_class = getattr(ccxt, self._exchange_name)
        self._exchange = self._create_exchange()

    @classmethod
    def create(cls, *, exchange_name: str, **kwargs: Any) -> Self:  # type: ignore[override]
        """Factory method to create a CCXT adapter instance.

        Args:
            exchange_name: Name of the exchange (e.g., 'binance', 'bitget')
            **kwargs: Additional CCXT-specific configuration options
        """
        return cls(exchange_name)

    async def __aenter__(self) -> Self:
        """Rebuild the exchange with a session, which CCXT only accepts at
        construction.
        """
        session = _build_session(self._exchange.cafile)
        exchange = self._create_exchange(session=session)
        try:
            await exchange.__aenter__()
            await exchange.load_markets()
        except BaseException:
            await exchange.close()
            await session.close()
            raise
        self._exchange = exchange
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the session `__aenter__` built, since the exchange does not own it."""
        session = self._exchange.session
        try:
            await self._exchange.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if session is not None:
                await session.close()

    def _get_supported_symbols(self) -> set[Symbol]:
        """The exchange's own symbols, minus formats `Symbol` cannot represent."""
        symbols: set[Symbol] = set()
        for s in self._exchange.symbols:
            try:
                symbols.add(Symbol.create(s))
            except ValueError:
                # Skip instruments not supported by our Symbol format (e.g., dated futures)
                continue
        return symbols

    def _get_supported_timeframes(self) -> set[TimeFrame]:
        return set(self._exchange.timeframes.keys())

    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_ms: int,
        end_ms: int,
        include_open_candle: bool = False,
    ) -> OhlcvData:
        """Fetch OHLCV data from the exchange.

        Args:
            start_ms: Start of the range in milliseconds
            end_ms: End of the range in milliseconds

        Returns:
            Candles of the range, the one in progress left out.

        Raises:
            InvalidSymbolError: If symbol is not supported by this exchange
            InvalidTimeframeError: If timeframe is not supported by this exchange
            DownloadError: If data cannot be fetched
        """
        if symbol not in self._get_supported_symbols():
            raise InvalidSymbolError(symbol)

        supported_timeframes = self._get_supported_timeframes() & set(TIMEFRAMES)
        if timeframe not in supported_timeframes:
            raise InvalidTimeframeError(timeframe, supported_timeframes)

        timeframe_ms = to_milliseconds(timeframe)
        limit = self._page_candles(timeframe_ms)
        rows = await fetch_pages(
            partial(self._fetch_page, symbol, timeframe, limit, timeframe_ms),
            page_bounds(start_ms, end_ms, (limit - 1) * timeframe_ms),
        )
        if include_open_candle:
            return rows
        return without_the_open_candle(rows, timeframe)

    def _page_candles(self, timeframe_ms: int) -> int:
        """A venue refusing a request that spans more than a set number of
        days holds a page to the candles those days cover, so no request a
        page makes spans more than the venue allows.
        """
        venue_config = self._venue_config()
        limit = int(venue_config["limit_size_request"])
        if "request_span_days" not in venue_config:
            return limit
        span_ms = int(venue_config["request_span_days"]) * _DAY_MS
        return min(limit, span_ms // timeframe_ms)

    def _venue_config(self) -> dict[str, Any]:
        return EXCHANGES.get(self._exchange_name, DEFAULT_EXCHANGE_CONFIG)

    async def _fetch_page(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        limit: int,
        timeframe_ms: int,
        page_start_ms: int,
        page_end_ms: int,
    ) -> OhlcvData:
        """A page holds one candle fewer than `limit` and is asked as the
        range from its start to the last millisecond of the candle after it,
        `limit` candles in all. A venue counting `limit` candles back from the
        range's end answers the page and the candle after it; one answering
        the candles closed inside the range, from a start CCXT may set a
        millisecond before the page and the venue floors to the candle before
        it, answers that candle and the page. Either way the page comes back
        whole, and the extra candle is a neighbouring page's.
        """
        try:
            page: list[list[float]] = await self._exchange.fetch_ohlcv(
                str(symbol),
                str(timeframe),
                page_start_ms,
                limit,
                {"until": page_end_ms + timeframe_ms - 1},
            )
        except ccxt.DDoSProtection as e:
            raise RateLimitExceededError(symbol, timeframe, str(e)) from e
        except ccxt.RateLimitExceeded as e:
            raise RateLimitExceededError(symbol, timeframe, str(e)) from e
        except ccxt.NetworkError as e:
            raise ExchangeConnectionError(symbol, timeframe, str(e)) from e
        except Exception as e:
            raise DownloadError(
                f"Failed to fetch {symbol}/{timeframe}: {str(e)}"
            ) from e

        if not page:
            return no_candles()
        return np.array(page, dtype=np.float64)

    def _create_exchange(self, session: aiohttp.ClientSession | None = None) -> Any:
        """Build the CCXT exchange, which takes a session only at construction.

        An interval is declared only where a venue grants fewer requests than
        CCXT expects.
        """
        options: dict[str, Any] = {"enableRateLimit": True}
        venue_config = self._venue_config()
        if "request_interval_ms" in venue_config:
            options["rateLimit"] = venue_config["request_interval_ms"]
        if session is not None:
            options["session"] = session
        return self._exchange_class(options)


def _build_session(cafile: str) -> aiohttp.ClientSession:
    """Resolve host names through the operating system, like the rest of the engine.

    Left to aiohttp, CCXT's aiodns dependency resolves through c-ares, which fails
    wherever the nameservers cannot be queried directly.

    Args:
        cafile: Certificate bundle CCXT selected, so supplying a session does not
            change how certificates are verified.
    """
    connector = aiohttp.TCPConnector(
        ssl=ssl.create_default_context(cafile=cafile),
        resolver=aiohttp.ThreadedResolver(),
        enable_cleanup_closed=True,
    )
    return aiohttp.ClientSession(connector=connector)
