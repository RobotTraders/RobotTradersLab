import time

import ccxt.async_support as ccxt_async
import numpy as np
import pytest

from robottraderslab._core import to_milliseconds

_MILLISECONDS_PER_SECOND = 1000


class _FakeCcxtBitget:
    """CCXT Bitget stand-in answering one of the venue's two candle endpoints
    the way the venue aligns a range, from the request CCXT builds out of
    `since`, `limit` and `params`, and handing back only the candles stamped
    from `since` on, the ones `fetch_ohlcv` ever returns. A request is held
    to the candles of ninety days, the span the venue allows one to cover.
    """

    cafile = None
    request_span_ms = 90 * 24 * 60 * 60 * _MILLISECONDS_PER_SECOND

    def __init__(self, options: dict) -> None:
        self.session = options.get("session")
        self.symbols = ["BTC/USDT:USDT", "SOL/USDT:USDT"]
        self.timeframes = {"1h": "1h", "4h": "4h", "1d": "1d"}

    async def __aenter__(self) -> "_FakeCcxtBitget":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def close(self) -> None:
        return None

    async def load_markets(self) -> dict:
        return {}

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
        params: dict,
    ) -> list[list[float]]:
        period_ms = to_milliseconds(timeframe)
        limit = min(limit, self.request_span_ms // period_ms)
        end_ms = min(
            params.get("until", since + limit * period_ms),
            int(time.time() * _MILLISECONDS_PER_SECOND),
        )
        stamps = self._answered_stamps(since, limit, period_ms, params, end_ms)
        return [
            [float(stamp), 1.0, 2.0, 0.5, 1.5, 10.0]
            for stamp in stamps[stamps >= since]
        ]

    def _answered_stamps(
        self, since: int, limit: int, period_ms: int, request: dict, end_ms: int
    ) -> np.ndarray:
        raise NotImplementedError


class FakeCcxtBitgetRecentEndpoint(_FakeCcxtBitget):
    """The recent endpoint ends its answer on the candle whose period holds
    `endTime` and reaches back over the candle periods the range spans, at
    most `limit`.
    """

    def _answered_stamps(
        self, since: int, limit: int, period_ms: int, request: dict, end_ms: int
    ) -> np.ndarray:
        last_stamp = end_ms // period_ms * period_ms
        count = min(limit, -(-(end_ms - since) // period_ms))
        return np.arange(
            last_stamp - (count - 1) * period_ms, last_stamp + 1, period_ms
        )


class FakeCcxtBitgetHistoryEndpoint(_FakeCcxtBitget):
    """The history endpoint ends its answer on the last candle closed by
    `endTime` and reaches back over `limit` candles.
    """

    def _answered_stamps(
        self, since: int, limit: int, period_ms: int, request: dict, end_ms: int
    ) -> np.ndarray:
        last_closed = (end_ms - period_ms) // period_ms * period_ms
        return np.arange(
            last_closed - (limit - 1) * period_ms, last_closed + 1, period_ms
        )


@pytest.fixture
def bitget_answering_from_its_recent_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ccxt_async, "bitget", FakeCcxtBitgetRecentEndpoint)


@pytest.fixture
def bitget_answering_from_its_history_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ccxt_async, "bitget", FakeCcxtBitgetHistoryEndpoint)
