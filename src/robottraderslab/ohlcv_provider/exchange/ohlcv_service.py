import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from robottraderslab._core import (
    DataError,
    OhlcvValidationError,
    Symbol,
    TimeFrame,
    candles_back,
    newest_settled_candle_start,
)

from ..date_utils import filter_date_range, standardize_ohlcv_index
from ..exceptions import PartialDownloadError
from ..interfaces import (
    OhlcvRepositoryProtocol,
)
from ..types import DateRange, OhlcvColumn
from .download import fetch_missing_data
from .download_metadata import (
    load_earliest_available,
    load_empty_ranges,
    store_earliest_available,
    store_empty_ranges,
)
from .gap_analysis import compute_storage_gaps, effective_window_start
from .missing_candles import MarketOpenMask, trades_between
from .ohlcv_adapter_factory import OhlcvAdapterFactory

logger = logging.getLogger(__name__)


async def fetch_ohlcv_with(
    adapter_factory: OhlcvAdapterFactory,
    repository: OhlcvRepositoryProtocol,
    adapter_name: str,
    dataset_name: str,
    symbol: Symbol,
    timeframe: TimeFrame,
    start_date: datetime,
    end_date: datetime,
    market_open_mask: MarketOpenMask,
    show_progress: bool = False,
    adapter_config: dict[str, Any] | None = None,
    storage_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data using the injected repository and exchange.

    Args:
        adapter_name: Name the adapter is registered under.
        dataset_name: Name the series is stored and remembered under.
        market_open_mask: The venue's calendar, as its adapter's class declares
            it.
        storage_dir: Storage directory for metadata persistence.

    Raises:
        OhlcvValidationError: If inputs are invalid.
        DownloadError: If every missing range failed on a transient error, or
            part of a download succeeded before another part failed.
        DataError: If every missing range failed on an error a retry cannot
            fix, or another data operation fails.
        ExchangeCriticalError: If the adapter reports the venue unusable.
        StrategyCriticalError: If the adapter reports a symbol the venue
            cannot trade; only the config can name one it can.
    """
    try:
        start_date, end_date = _validate_dates(start_date, end_date)

        earliest = (
            load_earliest_available(storage_dir, dataset_name, symbol, timeframe)
            if storage_dir
            else None
        )
        known_empty = (
            load_empty_ranges(storage_dir, dataset_name, symbol, timeframe)
            if storage_dir
            else []
        )
        existing_data = await repository.load(symbol, timeframe)

        missing_ranges = compute_storage_gaps(
            existing_data,
            start_date,
            end_date,
            timeframe,
            market_open_mask,
            earliest,
            known_empty,
        )

        if missing_ranges:
            earliest_start = missing_ranges[0][0]
            latest_end = missing_ranges[-1][1]
            logger.info(
                f"{symbol}@{timeframe}: downloading data for {earliest_start} - {latest_end}"
            )
            async with adapter_factory.using(
                adapter_name, **(adapter_config or {})
            ) as venue:
                try:
                    new_data = await fetch_missing_data(
                        venue,
                        symbol,
                        timeframe,
                        missing_ranges,
                        show_progress=show_progress,
                    )
                except PartialDownloadError as e:
                    await repository.store(symbol, timeframe, e.data)
                    raise
            if not new_data.empty:
                await repository.store(symbol, timeframe, new_data)
                result = filter_date_range(
                    _merged(existing_data, new_data), start_date, end_date
                )
            else:
                result = filter_date_range(existing_data, start_date, end_date)
            if storage_dir:
                _record_empty_ranges(
                    storage_dir,
                    dataset_name,
                    symbol,
                    timeframe,
                    missing_ranges,
                    _candles_added(existing_data, new_data),
                )
                _record_earliest_available(
                    storage_dir,
                    dataset_name,
                    symbol,
                    timeframe,
                    missing_ranges,
                    new_data,
                    effective_window_start(start_date, earliest),
                    market_open_mask,
                )
        else:
            result = filter_date_range(existing_data, start_date, end_date)

        return result

    except DataError:
        raise
    except Exception as e:
        raise DataError(
            f"Failed to retrieve OHLCV data for {symbol}/{timeframe}: {str(e)}"
        ) from e


def _merged(stored: pd.DataFrame, downloaded: pd.DataFrame) -> pd.DataFrame:
    if stored.empty:
        return standardize_ohlcv_index(downloaded)
    return standardize_ohlcv_index(pd.concat([stored, downloaded]))


def _candles_added(stored: pd.DataFrame, downloaded: pd.DataFrame) -> pd.DataFrame:
    """The range after the last stored candle is asked from that candle, so
    the exchange answers it whether or not it published the ones after it.
    """
    if stored.empty or downloaded.empty:
        return downloaded
    held = stored.index[stored[OhlcvColumn.CLOSE].notna()]
    return downloaded[~downloaded.index.isin(held)]


def _record_empty_ranges(
    storage_dir: Path,
    dataset_name: str,
    symbol: Symbol,
    timeframe: TimeFrame,
    requested: Sequence[DateRange],
    added: pd.DataFrame,
) -> None:
    """Remember closed periods the exchange answered as empty.

    A range ends on a stamp the exchange answers with whether or not it
    published the candles before it, the next stored candle or the one the
    range after it is asked from, so a range is empty when the candles the
    download adds to the storage hold none stamped before its end. A range
    that reaches into the candle currently forming is left out: it is empty
    only because that candle has not closed yet, and a period is remembered
    as empty only once the exchange has published all of it.
    """
    settled_before = newest_settled_candle_start(timeframe, datetime.now(timezone.utc))
    empty = [
        (range_start, range_end)
        for range_start, range_end in requested
        if range_end < settled_before
        and (
            added.empty
            or not ((added.index >= range_start) & (added.index < range_end)).any()
        )
    ]
    if not empty:
        return

    logger.info(
        f"{symbol}@{timeframe}: {len(empty)} range(s) hold no data at the exchange, "
        "not requesting them again"
    )
    try:
        store_empty_ranges(storage_dir, dataset_name, symbol, timeframe, empty)
    except OSError as e:
        logger.warning(
            "%s@%s: failed to write empty-range metadata: %s", symbol, timeframe, e
        )


def _record_earliest_available(
    storage_dir: Path,
    dataset_name: str,
    symbol: Symbol,
    timeframe: TimeFrame,
    requested: Sequence[DateRange],
    downloaded: pd.DataFrame,
    window_start: datetime,
    market_open_mask: MarketOpenMask,
) -> None:
    """Remember where the exchange's history begins, from its own answer.

    When the head of the window is requested and the answer starts a whole
    candle or more after it, the exchange skipped a candle it could have
    published, so its history begins where it answered. A request need not
    fall on the exchange's candle grid, and the first candle any exchange can
    answer with lies within one candle of it, so an answer that close proves
    nothing. No candles at all proves nothing either, since a transient venue
    failure answers the same way, and neither does a candle on the stamp the
    head range ends on, which is also where the range after it is asked from.
    A market closed for every stamp between the request and that first candle
    published none of them whatever its history holds, so its silence proves
    nothing. Gap analysis clamps every later window to whatever is recorded,
    so a boundary taken from any of them would never be probed again.
    """
    head_start, head_end = requested[0]
    if head_start != window_start or downloaded.empty:
        return
    boundary = downloaded.index.min().to_pydatetime()
    first_candle_after_head = pd.Timestamp(head_start) + candles_back(timeframe, 1)
    if boundary < first_candle_after_head or boundary >= head_end:
        return
    if not trades_between(market_open_mask, timeframe, head_start, boundary):
        return
    try:
        store_earliest_available(storage_dir, dataset_name, symbol, timeframe, boundary)
    except OSError as e:
        logger.warning(
            "%s@%s: failed to write earliest-available metadata: %s",
            symbol,
            timeframe,
            e,
        )


def _validate_dates(
    start_date: datetime | None, end_date: datetime | None
) -> tuple[datetime, datetime]:
    """Confirm both dates are present, valid and start comes before end.

    Raises:
        OhlcvValidationError: If either date is missing, not a datetime, or
            out of order.
    """
    if start_date is None:
        raise OhlcvValidationError("start_date is required")
    if end_date is None:
        raise OhlcvValidationError("end_date is required")

    if not isinstance(start_date, datetime):
        raise OhlcvValidationError("start_date must be a datetime object")
    if not isinstance(end_date, datetime):
        raise OhlcvValidationError("end_date must be a datetime object")

    if end_date <= start_date:
        raise OhlcvValidationError("end_date must be after start_date")

    return start_date, end_date
