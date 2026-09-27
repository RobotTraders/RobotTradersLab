import asyncio
import logging
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from robottraderslab._core import (
    TIMEFRAMES,
    DataError,
    DownloadError,
    ExchangeTransientError,
    OhlcvValidationError,
    Symbol,
    TimeFrame,
    drain_tasks,
)
from robottraderslab.exchanges import OhlcvAdapterProtocol, OhlcvData

from ..date_utils import (
    filter_date_range,
    standardize_ohlcv_index_from_timestamp,
)
from ..exceptions import PartialDownloadError
from ..types import DateRanges, OhlcvColumn
from .missing_candles import MarketOpenMask, handle_missing_candles

logger = logging.getLogger(__name__)


async def fetch_missing_data(
    exchange: OhlcvAdapterProtocol,
    symbol: Symbol,
    timeframe: TimeFrame,
    missing_ranges: DateRanges,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Fetch every missing range concurrently, keeping what arrives if one fails.

    Raises:
        PartialDownloadError: If a range failed after another had already
            arrived, carrying the data that did arrive.
        DownloadError: If every range failed on a transient error.
        DataError: If every range failed on an error a retry cannot fix.
        ExchangeCriticalError: If the adapter reports the venue unusable.
        StrategyCriticalError: If the adapter reports a symbol the venue
            cannot trade; only the config can name one it can.
    """
    fetched: list[pd.DataFrame] = []

    async def fetch_range_into(range_start: datetime, range_end: datetime) -> None:
        fetched.append(
            await _fetch_range(exchange, symbol, timeframe, range_start, range_end)
        )

    tasks = [
        asyncio.create_task(fetch_range_into(range_start, range_end))
        for range_start, range_end in missing_ranges
    ]
    try:
        await _await_with_progress(
            tasks, f"Downloading {symbol} {timeframe}", show_progress
        )
    except Exception as failure:
        await drain_tasks(tasks)
        arrived = _combine(fetched)
        if arrived.empty:
            raise
        raise PartialDownloadError(
            f"{symbol}@{timeframe}: keeping {len(arrived)} candles "
            f"fetched before the download failed: {failure}",
            arrived,
        ) from failure
    except BaseException:
        await drain_tasks(tasks)
        raise

    return _combine(fetched)


def require_supported(timeframe: TimeFrame) -> None:
    """Refuse a timeframe before it reaches the venue, so a venue only ever
    receives one the framework knows how to page and stamp.

    Raises:
        OhlcvValidationError: If the timeframe is unknown to the framework.
    """
    if timeframe not in TIMEFRAMES:
        raise OhlcvValidationError(
            f"Unsupported timeframe: {timeframe}. Supported: {', '.join(TIMEFRAMES)}"
        )


def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=False)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


async def _await_with_progress(
    tasks: list[asyncio.Task], desc: str, show_progress: bool
) -> None:
    with tqdm(total=len(tasks), desc=desc, disable=not show_progress) as pbar:
        for completed in asyncio.as_completed(tasks):
            await completed
            pbar.update(1)


async def _fetch_range(
    exchange: OhlcvAdapterProtocol,
    symbol: Symbol,
    timeframe: TimeFrame,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    require_supported(timeframe)

    rows = await _fetch_rows(
        exchange,
        symbol,
        timeframe,
        int(start_date.timestamp() * 1000),
        int(end_date.timestamp() * 1000),
    )
    if rows.size == 0:
        logger.warning(
            f"{symbol}@{timeframe}: no data found in {start_date} - {end_date}"
        )
        return pd.DataFrame()

    return _clean_and_window(
        rows, symbol, timeframe, start_date, end_date, exchange.market_open_mask
    )


async def _fetch_rows(
    exchange: OhlcvAdapterProtocol,
    symbol: Symbol,
    timeframe: TimeFrame,
    start_ms: int,
    end_ms: int,
) -> OhlcvData:
    try:
        return await exchange.fetch_ohlcv(symbol, timeframe, start_ms, end_ms)
    except ExchangeTransientError as e:
        raise DownloadError(
            f"{symbol}@{timeframe}: failed to fetch from exchange: {str(e)}"
        ) from e
    except Exception as e:
        raise DataError(
            f"{symbol}@{timeframe}: failed to fetch from exchange: {str(e)}"
        ) from e


def _clean_and_window(
    rows: OhlcvData,
    symbol: Symbol,
    timeframe: TimeFrame,
    start_date: datetime,
    end_date: datetime,
    market_open_mask: MarketOpenMask,
) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(OhlcvColumn))
    df = df.sort_values(OhlcvColumn.TIMESTAMP)
    df = df.drop_duplicates(subset=OhlcvColumn.TIMESTAMP, keep="first")
    df = handle_missing_candles(df, symbol, timeframe, market_open_mask)
    df = standardize_ohlcv_index_from_timestamp(df)
    return filter_date_range(df, start_date, end_date)
