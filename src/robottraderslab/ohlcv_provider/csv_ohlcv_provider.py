import io
from datetime import datetime
from pathlib import Path

import pandas as pd

from robottraderslab._core import (
    DataError,
    OHLCVProviderProtocol,
    OhlcvValidationError,
    Symbol,
    SymbolStr,
    TimeFrame,
)

from .date_utils import (
    filter_date_range,
    standardize_ohlcv_index,
    to_datetime,
)


class CSVOHLCVProvider(OHLCVProviderProtocol):
    """Raw OHLCV provider that loads a single symbol/timeframe from a CSV file."""

    def __init__(
        self,
        file: str | Path | io.StringIO,
        symbol: SymbolStr | Symbol,
        timeframe: TimeFrame,
    ) -> None:
        self._file_or_buffer = file
        self._symbol = Symbol.create(symbol)
        self._timeframe = timeframe
        self._start_dt: datetime | None = None
        self._end_dt: datetime | None = None
        self._lookback: int = 0
        self._ohlcv: pd.DataFrame | None = None

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
        """Extend start_dt backwards to provide warmup candles for rolling indicators.

        In backtest mode (explicit dates), shifts the CSV data window back so that
        the strategy's rolling windows are fully populated at start_date.

        Args:
            lookbacks: Per-timeframe candle counts declared by the strategy.
        """
        lookback = lookbacks.get(self._timeframe, 0)
        if lookback > 0:
            self._lookback = lookback
            self._ohlcv = None

    def fetch_ohlcv(self, symbol: Symbol, timeframe: TimeFrame) -> pd.DataFrame:
        if symbol != self._symbol or timeframe != self._timeframe:
            raise OhlcvValidationError(
                "CSV provider configured for a single symbol/timeframe"
            )

        return self._get_cached_ohlcv()

    def get_all_cached_ohlcv_for_symbol(self, symbol: Symbol) -> pd.DataFrame:
        """Get all cached OHLCV for a symbol."""
        if symbol != self._symbol:
            raise OhlcvValidationError(
                f"CSV provider configured for {self._symbol}, not {symbol}"
            )
        return self._get_cached_ohlcv()

    def _get_cached_ohlcv(self) -> pd.DataFrame:
        """Return cached OHLCV, loading once on first access."""
        if self._ohlcv is None:
            self._ohlcv = self._load_data()
        return self._ohlcv

    def _load_data(self) -> pd.DataFrame:
        try:
            ohlcv = pd.read_csv(
                self._file_or_buffer, parse_dates=["date"], index_col="date"
            )
        except Exception as e:
            raise DataError(f"Failed reading CSV {self._file_or_buffer}: {e}") from e
        if ohlcv.empty:
            raise DataError(f"No data available in CSV file: {self._file_or_buffer}")

        ohlcv = standardize_ohlcv_index(ohlcv)
        data_start_dt = self._resolve_data_start(ohlcv)
        ohlcv = filter_date_range(ohlcv, data_start_dt, self._end_dt)
        if (self._start_dt is not None or self._end_dt is not None) and ohlcv.empty:
            raise DataError(f"No data found between {data_start_dt} and {self._end_dt}")
        return ohlcv

    def _resolve_data_start(self, ohlcv: pd.DataFrame) -> datetime | None:
        if self._start_dt is None or self._lookback <= 0:
            return self._start_dt
        before_start = ohlcv.index[ohlcv.index < self._start_dt]
        if before_start.empty:
            return self._start_dt
        warmup_start: pd.Timestamp = before_start[-self._lookback :][0]
        return warmup_start.to_pydatetime()
