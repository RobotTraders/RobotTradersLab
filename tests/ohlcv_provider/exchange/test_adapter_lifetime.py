import asyncio
import logging
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab.exceptions import ExchangeCriticalError
from robottraderslab.exchanges import OhlcvData, to_milliseconds
from robottraderslab.ohlcv_provider import ExchangeOHLCVProvider
from robottraderslab.ohlcv_provider.exceptions import PartialDownloadError
from robottraderslab.ohlcv_provider.exchange.ohlcv_adapter_factory import (
    OhlcvAdapterFactory,
)
from robottraderslab.ohlcv_provider.exchange.ohlcv_cache import _OHLCV_CACHE

_ADAPTER_NAME = "counting_exchange"
_TIMEFRAME: TimeFrame = "1h"
_START = "2025-01-01"
_END = "2025-01-03"
_SLOW_SYMBOL = Symbol.create("ETH/USDT:USDT")
_FAST_SYMBOL = Symbol.create("BTC/USDT:USDT")
_DROPPED_CONNECTION = OSError("the venue dropped the connection being closed")


class _CountingAdapter:
    close_failure: Exception | None = None

    def __init__(self) -> None:
        self.opens = 0
        self.closes = 0
        self.fetches_while_closed: list[Symbol] = []
        self.slow_symbol_fetching = asyncio.Event()
        self.slow_symbol_may_finish = asyncio.Event()

    @classmethod
    def dataset_qualifiers(cls, **_kwargs: Any) -> tuple[str, ...]:
        return ()

    async def __aenter__(self) -> "_CountingAdapter":
        await asyncio.sleep(0)
        self.opens += 1
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.closes += 1
        if self.close_failure is not None:
            raise self.close_failure

    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_ms: int,
        end_ms: int,
        include_open_candle: bool = False,
    ) -> OhlcvData:
        if symbol == _SLOW_SYMBOL:
            self.slow_symbol_fetching.set()
            await self.slow_symbol_may_finish.wait()
        else:
            await self.slow_symbol_fetching.wait()
        if self.closes:
            self.fetches_while_closed.append(symbol)
        return _candles(timeframe, start_ms, end_ms)

    @classmethod
    def market_open_mask(cls, index: pd.DatetimeIndex) -> np.ndarray | None:
        return None


def _candles(timeframe: TimeFrame, start_ms: int, end_ms: int) -> OhlcvData:
    step = to_milliseconds(timeframe)
    stamps = np.arange(start_ms - start_ms % step + step, end_ms, step, dtype=np.int64)
    rows = np.empty((stamps.size, 6), dtype=np.float64)
    rows[:, 0] = stamps
    rows[:, 1:] = 1.0
    return rows


@pytest.fixture
def built_adapters(
    make_registered_adapter: Callable[..., None],
) -> list[_CountingAdapter]:
    built: list[_CountingAdapter] = []

    class _Registered(_CountingAdapter):
        @classmethod
        def create(cls, **_kwargs: Any) -> "_Registered":
            adapter = cls()
            built.append(adapter)
            return adapter

    make_registered_adapter(_Registered, _ADAPTER_NAME)
    return built


@pytest.fixture(autouse=True)
def _empty_candle_cache() -> Iterator[None]:
    _OHLCV_CACHE.clear()
    yield
    _OHLCV_CACHE.clear()


@pytest.fixture
def provider() -> ExchangeOHLCVProvider:
    built = ExchangeOHLCVProvider(exchange=_ADAPTER_NAME)
    built.set_dates(_START, _END)
    return built


async def _download_both_symbols(
    provider: ExchangeOHLCVProvider, built_adapters: list[_CountingAdapter]
) -> None:
    """Fetch the two symbols the way the assembly layer does, with the fast
    symbol's download ending while the slow symbol is still fetching.
    """
    slow = asyncio.create_task(provider.fetch_ohlcvs([(_SLOW_SYMBOL, _TIMEFRAME)]))
    await provider.fetch_ohlcvs([(_FAST_SYMBOL, _TIMEFRAME)])
    built_adapters[-1].slow_symbol_may_finish.set()
    await slow


class TestAdapterAcrossRuns:
    def test_a_second_run_opens_an_adapter_of_its_own(self, built_adapters, provider):
        asyncio.run(_download_both_symbols(provider, built_adapters))
        _OHLCV_CACHE.clear()

        asyncio.run(_download_both_symbols(provider, built_adapters))

        assert [adapter.opens for adapter in built_adapters] == [1, 1]

    def test_a_second_run_opens_its_own_after_a_close_fails(
        self, built_adapters, provider, monkeypatch
    ):
        monkeypatch.setattr(_CountingAdapter, "close_failure", _DROPPED_CONNECTION)
        asyncio.run(_download_both_symbols(provider, built_adapters))
        monkeypatch.setattr(_CountingAdapter, "close_failure", None)
        _OHLCV_CACHE.clear()

        asyncio.run(_download_both_symbols(provider, built_adapters))

        assert [adapter.opens for adapter in built_adapters] == [1, 1]


class TestAdapterSharedByConcurrentDownloads:
    async def test_the_shared_adapter_is_opened_once(self, built_adapters, provider):
        await _download_both_symbols(provider, built_adapters)

        assert [adapter.opens for adapter in built_adapters] == [1]

    async def test_no_symbol_fetches_after_the_adapter_is_closed(
        self, built_adapters, provider
    ):
        await _download_both_symbols(provider, built_adapters)

        assert built_adapters[0].fetches_while_closed == []

    async def test_the_adapter_is_closed_when_the_downloads_end(
        self, built_adapters, provider
    ):
        await _download_both_symbols(provider, built_adapters)

        assert built_adapters[0].closes == 1


class TestAFailedClose:
    async def test_the_download_keeps_its_own_outcome(
        self, built_adapters, monkeypatch, caplog
    ):
        monkeypatch.setattr(_CountingAdapter, "close_failure", _DROPPED_CONNECTION)
        factory = OhlcvAdapterFactory()
        arrived = pd.DataFrame({"close": [99.0]})

        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(PartialDownloadError) as failure,
        ):
            async with factory.using(_ADAPTER_NAME):
                raise PartialDownloadError("the venue stopped answering", arrived)

        assert failure.value.data is arrived
        assert built_adapters[0].closes == 1
        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]
        assert warnings == [
            f"OHLCV adapter `{_ADAPTER_NAME}` failed to close once its downloads ended"
        ]

    async def test_a_venue_that_became_unusable_stops_the_run(
        self, built_adapters, monkeypatch
    ):
        monkeypatch.setattr(
            _CountingAdapter,
            "close_failure",
            ExchangeCriticalError("the venue revoked the key"),
        )
        factory = OhlcvAdapterFactory()

        with pytest.raises(ExchangeCriticalError, match="revoked the key"):
            async with factory.using(_ADAPTER_NAME):
                pass
