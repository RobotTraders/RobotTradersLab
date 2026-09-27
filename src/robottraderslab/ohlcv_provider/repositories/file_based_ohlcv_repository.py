import contextlib
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd

from robottraderslab._core import DataError, Symbol, TimeFrame

from ..date_utils import filter_date_range, standardize_ohlcv_index

logger = logging.getLogger(__name__)


class FileBasedOhlcvRepository(ABC):
    """ABC for file-based OHLCV repositories."""

    def __init__(self, exchange_name: str, base_path: str | Path):
        """Initialise file-based repository for a specific exchange.

        Args:
            exchange_name: Name of the exchange (e.g., 'binance', 'ccxt_bitget').

        Raises:
            DataError: If base directory cannot be created.
        """
        self.exchange_name = exchange_name
        self._normalized_exchange_name = self._normalize_exchange_name(exchange_name)
        self._base_path = Path(base_path)

        try:
            self._base_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise DataError(
                f"Error creating storage directory {base_path}: {str(e)}"
            ) from e

    @property
    @abstractmethod
    def _file_extension(self) -> str: ...

    @abstractmethod
    def _read_file(self, file_path: Path) -> pd.DataFrame: ...

    @abstractmethod
    def _write_file(self, data: pd.DataFrame, file_path: Path) -> None: ...

    async def store(
        self, symbol: Symbol, timeframe: TimeFrame, data: pd.DataFrame
    ) -> None:
        """Store OHLCV data for a symbol/timeframe, merging with what is already there.

        Written to a temporary file first, so a crash mid-write never corrupts
        the existing cache.
        """
        if data.empty:
            return

        try:
            file_path = self._build_file_path(symbol, timeframe)
            data = standardize_ohlcv_index(data)

            if file_path.exists():
                existing_data = self._read_file(file_path)
                data = pd.concat([existing_data, data])
                data = data[~data.index.duplicated(keep="last")].sort_index()

            fd, tmp_path = tempfile.mkstemp(dir=file_path.parent, suffix=".tmp")
            try:
                os.close(fd)
                self._write_file(data, Path(tmp_path))
                os.replace(tmp_path, file_path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise

        except Exception as e:
            raise DataError(
                f"Error storing data for {str(symbol)}/{str(timeframe)}: {str(e)}"
            ) from e

    async def load(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        try:
            file_path = self._build_file_path(symbol, timeframe)

            if not file_path.exists():
                return pd.DataFrame()

            if file_path.stat().st_size == 0:
                logger.warning(
                    "Corrupted cache file (empty) for %s/%s, discarding: %s",
                    symbol,
                    timeframe,
                    file_path,
                )
                file_path.unlink()
                return pd.DataFrame()

            try:
                data = self._read_file(file_path)
            except (ValueError, TypeError):
                logger.warning(
                    "Corrupted cache file (unparseable) for %s/%s, discarding: %s",
                    symbol,
                    timeframe,
                    file_path,
                )
                file_path.unlink()
                return pd.DataFrame()

            if start_date or end_date:
                data = filter_date_range(data, start_date, end_date)

            return data

        except Exception as e:
            raise DataError(
                f"Error loading data for {str(symbol)}/{str(timeframe)}: {str(e)}"
            ) from e

    async def exists(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> bool:
        """Check if OHLCV data exists and optionally covers the requested date range."""
        try:
            file_path = self._build_file_path(symbol, timeframe)

            if not file_path.exists():
                return False

            if not start_date and not end_date:
                return True

            data = self._read_file(file_path)
            if data.empty:
                return False

            data_start = data.index.min()
            data_end = data.index.max()

            if start_date and data_start > start_date:
                return False
            if end_date and data_end < end_date:
                return False

            return True

        except Exception as e:
            raise DataError(
                f"Error checking existence for {str(symbol)}/{str(timeframe)}: {str(e)}"
            ) from e

    async def delete(self, symbol: Symbol, timeframe: TimeFrame) -> None:
        try:
            file_path = self._build_file_path(symbol, timeframe)

            if file_path.exists():
                file_path.unlink()

        except Exception as e:
            raise DataError(
                f"Error deleting data for {str(symbol)}/{str(timeframe)}: {str(e)}"
            ) from e

    def _build_file_path(self, symbol: Symbol, timeframe: TimeFrame) -> Path:
        timeframe_path = (
            self._base_path / self._normalized_exchange_name / str(timeframe)
        )
        timeframe_path.mkdir(parents=True, exist_ok=True)

        file_name = f"{self._format_symbol(str(symbol))}{self._file_extension}"
        return timeframe_path / file_name

    @staticmethod
    def _format_symbol(symbol: str) -> str:
        return symbol.replace("/", "-").replace(":", "-")

    @staticmethod
    def _normalize_exchange_name(exchange_name: str) -> str:
        return exchange_name.removeprefix("ccxt_")
