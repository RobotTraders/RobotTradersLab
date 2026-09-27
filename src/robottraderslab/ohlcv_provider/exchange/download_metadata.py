import json
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from robottraderslab._core import Symbol, TimeFrame

from ..types import DateRange, DateRanges

logger = logging.getLogger(__name__)


def load_earliest_available(
    storage_dir: Path, exchange: str, symbol: Symbol, timeframe: TimeFrame
) -> datetime | None:
    """Avoid re-downloading data the exchange does not have.

    Args:
        exchange: Exchange name (e.g., "bitget", "ccxt_binance").

    Returns:
        The earliest timestamp the exchange has data for, or None if unknown.
    """
    path = _meta_path(storage_dir, exchange, symbol, timeframe)
    stored = _read(path).get("earliest_available")
    if stored is None:
        return None
    try:
        return datetime.fromisoformat(stored)
    except (TypeError, ValueError) as e:
        logger.warning("Dropping unusable `earliest_available` in %s: %s", path, e)
        return None


def load_empty_ranges(
    storage_dir: Path, exchange: str, symbol: Symbol, timeframe: TimeFrame
) -> DateRanges:
    """Avoid re-downloading closed periods the exchange answered as empty.

    Args:
        exchange: Exchange name (e.g., "bitget", "ccxt_binance").

    Returns:
        The ranges known to hold no data at the exchange.
    """
    path = _meta_path(storage_dir, exchange, symbol, timeframe)
    stored = _read(path).get("empty_ranges")
    if stored is None:
        return []
    if not isinstance(stored, list):
        logger.warning("Dropping unusable `empty_ranges` in %s: %r", path, stored)
        return []
    ranges: DateRanges = []
    for entry in stored:
        try:
            start, end = entry
            ranges.append((datetime.fromisoformat(start), datetime.fromisoformat(end)))
        except (TypeError, ValueError) as e:
            logger.warning("Dropping unusable empty range %r in %s: %s", entry, path, e)
    return ranges


def store_earliest_available(
    storage_dir: Path,
    exchange: str,
    symbol: Symbol,
    timeframe: TimeFrame,
    earliest: datetime,
) -> None:
    """Persist the exchange's data boundary so gap analysis skips unfillable ranges.

    Args:
        exchange: Exchange name (e.g., "bitget", "ccxt_binance").
        earliest: The earliest timestamp the exchange has data for.
    """
    _write(
        storage_dir,
        exchange,
        symbol,
        timeframe,
        {"earliest_available": earliest.isoformat()},
    )


def store_empty_ranges(
    storage_dir: Path,
    exchange: str,
    symbol: Symbol,
    timeframe: TimeFrame,
    ranges: Sequence[DateRange],
) -> None:
    """Persist closed periods the exchange holds no data for.

    Deleting the metadata file forces a fresh download, the only way to
    recover data an exchange backfills into a range once recorded as empty.

    Args:
        exchange: Exchange name (e.g., "bitget", "ccxt_binance").
        ranges: Ranges to add to the ones already recorded.
    """
    known = load_empty_ranges(storage_dir, exchange, symbol, timeframe)
    merged = list(dict.fromkeys(known + list(ranges)))
    _write(
        storage_dir,
        exchange,
        symbol,
        timeframe,
        {
            "empty_ranges": [
                [start.isoformat(), end.isoformat()] for start, end in merged
            ]
        },
    )


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        content = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        logger.warning("Corrupted metadata file %s, deleting: %s", path, e)
        path.unlink(missing_ok=True)
        return {}
    return content if isinstance(content, dict) else {}


def _write(
    storage_dir: Path,
    exchange: str,
    symbol: Symbol,
    timeframe: TimeFrame,
    fields: dict[str, Any],
) -> None:
    path = _meta_path(storage_dir, exchange, symbol, timeframe)
    content = _read(path) | fields
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


def _meta_path(
    storage_dir: Path, exchange: str, symbol: Symbol, timeframe: TimeFrame
) -> Path:
    exchange_name = exchange.removeprefix("ccxt_")
    safe_symbol = str(symbol).replace("/", "-").replace(":", "-")
    return storage_dir / exchange_name / str(timeframe) / f"{safe_symbol}.meta.json"
