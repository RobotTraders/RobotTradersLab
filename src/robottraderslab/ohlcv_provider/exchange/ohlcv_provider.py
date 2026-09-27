import asyncio
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from robottraderslab._core import (
    OHLCVProviderProtocol,
    StrategyCriticalError,
    Symbol,
    TimeFrame,
    candle_stamp_before,
    candle_stamps,
    candles_back,
    drain_tasks,
    newest_settled_candle_start,
    run_async,
)

from ..date_utils import to_datetime
from ..exceptions import StorageTypeError
from ..interfaces import (
    OhlcvRepositoryProtocol,
)
from ..repositories.csv_ohlcv_repository import CsvOhlcvRepository
from ..repositories.null_ohlcv_repository import NullOhlcvRepository
from ..repositories.parquet_ohlcv_repository import ParquetOhlcvRepository
from .download import require_supported
from .missing_candles import MarketOpenMask
from .ohlcv_adapter_factory import (
    OhlcvAdapterFactory,
    dataset_name,
    market_open_mask_of,
)
from .ohlcv_cache import (
    get_all_cached_ohlcv_for_symbol,
    get_cached_ohlcv,
    store_ohlcv_in_cache,
)
from .ohlcv_service import fetch_ohlcv_with

logger = logging.getLogger(__name__)

_ONE_WEEK_MS = 7 * 24 * 60 * 60 * 1000
_BOOKED_CANDLE = 1


class ExchangeOHLCVProvider(OHLCVProviderProtocol):
    """OHLCV data provider that fetches from exchanges with caching and gap filling.

    Supports two modes:
    1. Explicit dates: Provide start_date and end_date (backtest mode)
    2. Strategy lookbacks: Call set_required_lookbacks() with per-timeframe requirements (live mode)
    """

    def __init__(
        self,
        *,
        exchange: str,
        storage_dir: str | Path | None = None,
        storage: dict[str, Any] | None = None,
        **adapter_config: Any,
    ) -> None:
        """Initialise the OHLCV provider.

        Args:
            exchange: Exchange name (e.g., 'bitget', 'ccxt_binance')
            storage_dir: Directory caching OHLCV data as parquet.
            storage: Storage backend config dict with 'type' and backend-specific keys.
                When neither is provided, a no-op repository is used (no disk persistence).
            **adapter_config: Additional adapter configuration.

        Raises:
            OhlcvValidationError: If configuration is invalid.
            StorageTypeError: If `storage`'s `type` is not recognised.
        """
        self._adapter_name = exchange
        self._dataset_name = dataset_name(exchange, adapter_config)
        self._lookback_by_timeframe: dict[TimeFrame, int] = {}
        self._adapter_config: dict[str, Any] = adapter_config
        self._market_open_mask = market_open_mask_of(exchange)

        self._start_dt: datetime | None = None
        self._end_dt: datetime | None = None

        self._storage_dir: Path | None = Path(storage_dir) if storage_dir else None
        storage_config = _resolve_storage_config(storage, storage_dir)
        storage_type = storage_config.pop("type")
        self._repository = _create_repository(
            storage_type, self._dataset_name, **storage_config
        )
        self._adapter_factory = OhlcvAdapterFactory()

    def fetch_ohlcv(self, symbol: Symbol, timeframe: TimeFrame) -> pd.DataFrame:
        start_dt, end_dt = self._resolve_date_range(symbol, timeframe)

        cached_data = get_cached_ohlcv(
            self._dataset_name, symbol, timeframe, start_dt, end_dt
        )
        if cached_data is not None:
            return cached_data

        return run_async(self._load_and_cache(symbol, timeframe, start_dt, end_dt))

    async def fetch_ohlcvs(
        self, symbol_timeframes: Iterable[tuple[Symbol, TimeFrame]]
    ) -> dict[tuple[Symbol, TimeFrame], pd.DataFrame]:
        """Fetch all pairs concurrently; duplicates are fetched once.

        Returns:
            OHLCV frame per pair.
        """
        pairs = list(dict.fromkeys(symbol_timeframes))
        tasks = [
            asyncio.create_task(self._fetch_one(symbol, timeframe))
            for symbol, timeframe in pairs
        ]
        try:
            frames = await asyncio.gather(*tasks)
        except BaseException:
            await drain_tasks(tasks)
            raise
        return dict(zip(pairs, frames))

    def get_all_cached_ohlcv_for_symbol(self, symbol: Symbol) -> pd.DataFrame:
        """Get all cached OHLCV for a symbol merged across all timeframes.

        Returns:
            Sorted by timestamp, de-duplicated across timeframes.
        """
        return get_all_cached_ohlcv_for_symbol(self._dataset_name, symbol)

    def set_dates(self, start_date: str, end_date: str) -> None:
        """
        Set dates for data loading.

        Args:
            start_date: Start date (ISO format or datetime string)
            end_date: End date (ISO format or datetime string)
        """
        self._start_dt = to_datetime(start_date)
        self._end_dt = to_datetime(end_date)

    def set_required_lookbacks(self, lookbacks: dict[TimeFrame, int]) -> None:
        """A lookback is the history before the candle a run books on: in live
        mode (no explicit dates) it reaches back from the newest settled
        candle, in backtest mode (explicit dates) from start_dt, so rolling
        windows have enough historical data under either.

        Raises:
            OhlcvValidationError: If a timeframe is unknown to the framework.
        """
        for timeframe in lookbacks:
            require_supported(timeframe)
        self._lookback_by_timeframe = lookbacks.copy()
        self._lookback_reference_time = datetime.now(timezone.utc)

    async def _fetch_one(self, symbol: Symbol, timeframe: TimeFrame) -> pd.DataFrame:
        start_dt, end_dt = self._resolve_date_range(symbol, timeframe)
        cached = get_cached_ohlcv(
            self._dataset_name, symbol, timeframe, start_dt, end_dt
        )
        if cached is not None:
            return cached
        return await self._load_and_cache(symbol, timeframe, start_dt, end_dt)

    async def _load_and_cache(
        self, symbol: Symbol, timeframe: TimeFrame, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        result = await self._load(symbol, timeframe, start_dt, end_dt)
        store_ohlcv_in_cache(
            self._dataset_name,
            symbol,
            timeframe,
            start_dt,
            end_dt,
            result,
        )
        return result

    async def _load(
        self, symbol: Symbol, timeframe: TimeFrame, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        return await fetch_ohlcv_with(
            adapter_factory=self._adapter_factory,
            repository=self._repository,
            adapter_name=self._adapter_name,
            dataset_name=self._dataset_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_dt,
            end_date=end_dt,
            market_open_mask=self._market_open_mask,
            adapter_config=self._adapter_config,
            storage_dir=self._storage_dir,
        )

    def _resolve_date_range(
        self, symbol: Symbol, timeframe: TimeFrame
    ) -> tuple[datetime, datetime]:
        """A declared lookback is warm-up before the candle a run books on
        under either mode, so an indicator reading a candle against the one
        before it holds both values. Both bounds are candle stamps in live
        mode, so every symbol fetched in the cycle asks for the same window
        throughout a candle. The lookback counts the candles the venue's own
        calendar publishes.
        """
        if self._start_dt is not None and self._end_dt is not None:
            start_dt = self._start_dt
            lookback = self._lookback_by_timeframe.get(timeframe)
            if lookback is not None and lookback > 0:
                start_dt = _published_lookback_start(
                    candle_stamp_before(timeframe, self._start_dt),
                    timeframe,
                    lookback,
                    self._market_open_mask,
                    symbol,
                )
            return start_dt, self._end_dt

        if self._lookback_by_timeframe:
            lookback = self._lookback_by_timeframe.get(timeframe)
            if lookback is not None and lookback > 0:
                end_dt = newest_settled_candle_start(
                    timeframe, self._lookback_reference_time
                )
                start_dt = _published_lookback_start(
                    end_dt,
                    timeframe,
                    lookback + _BOOKED_CANDLE,
                    self._market_open_mask,
                    symbol,
                )
                return start_dt, end_dt

        return cast(datetime, self._start_dt), cast(datetime, self._end_dt)


def _resolve_storage_config(
    storage: dict[str, Any] | None,
    storage_dir: str | Path | None,
) -> dict[str, Any]:
    if storage is not None:
        config = dict(storage)
        config.setdefault("type", "parquet")
        return config
    if storage_dir is not None:
        return {"type": "parquet", "dir": storage_dir}
    return {"type": "none"}


def _create_repository(
    storage_type: str,
    exchange_name: str,
    **storage_config: Any,
) -> OhlcvRepositoryProtocol:
    match storage_type.lower():
        case "csv":
            return CsvOhlcvRepository(exchange_name, storage_config["dir"])
        case "parquet":
            return ParquetOhlcvRepository(exchange_name, storage_config["dir"])
        case "none":
            return NullOhlcvRepository()
        case _:
            raise StorageTypeError(storage_type)


def _published_lookback_start(
    last_dt: datetime,
    timeframe: TimeFrame,
    lookback: int,
    market_open_mask: MarketOpenMask,
    symbol: Symbol,
) -> datetime:
    """The stamp `lookback` published candles reach back to from `last_dt`,
    which counts as one of them when the calendar has it open. A closed stamp
    is not one the venue published a candle for, so it does not count toward
    `lookback`.

    Raises:
        StrategyCriticalError: If the mask reports no open stamp across a
            span longer than a week, since no venue calendar the engine
            ships closes that long.
    """
    last_ms = _to_epoch_ms(last_dt)
    reach = lookback
    while True:
        candidate_start_ms = _to_epoch_ms(
            pd.Timestamp(last_dt) - candles_back(timeframe, reach - 1)
        )
        stamps = candle_stamps(timeframe, candidate_start_ms, last_ms)
        open_mask = market_open_mask(pd.to_datetime(stamps, unit="ms", utc=True))
        if open_mask is None:
            return _from_epoch_ms(candidate_start_ms)
        open_positions = np.flatnonzero(open_mask)
        if open_positions.size >= lookback:
            return _from_epoch_ms(int(stamps[open_positions[-lookback]]))
        if open_positions.size == 0 and last_ms - candidate_start_ms > _ONE_WEEK_MS:
            raise StrategyCriticalError(
                f"{symbol}@{timeframe}: no candle published between "
                f"{_from_epoch_ms(candidate_start_ms)} and {_from_epoch_ms(last_ms)}"
            )
        reach *= 2


def _to_epoch_ms(moment: datetime | pd.Timestamp) -> int:
    return int(pd.Timestamp(moment).value // 1_000_000)


def _from_epoch_ms(stamp_ms: int) -> datetime:
    return pd.Timestamp(stamp_ms, unit="ms", tz="UTC").to_pydatetime()
