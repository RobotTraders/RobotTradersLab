from dataclasses import dataclass
from datetime import datetime
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..symbol import Symbol
from ..timeframes import TimeFrame

type _SymbolData = dict[str, np.ndarray]


class OHLCVRow(NamedTuple):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


OHLCVsBySymbol = dict[Symbol, OHLCVRow]


@dataclass(frozen=True, slots=True)
class TimeframeSnapshot:
    timestamp: datetime
    triggered_timeframes: list[TimeFrame]
    _symbol_data_by_tf: dict[TimeFrame, dict[Symbol, _SymbolData]]
    _close_indices_by_tf: dict[TimeFrame, pd.DatetimeIndex]
    _row_indices_by_tf: dict[TimeFrame, npt.NDArray[np.intp]]
    _iteration_index: int
    _stepping_rows: dict[tuple[TimeFrame, Symbol], int]

    @property
    def ohlcvs_by_symbol(self) -> OHLCVsBySymbol:
        """Computed on each access, since only the backtester reads it.

        Each row is stamped with its candle's close, the moment a fill on it
        is reported at. A symbol is stepped on its shortest declared
        timeframe's candles alone.
        """
        ohlcvs_by_symbol: OHLCVsBySymbol = {}

        for timeframe in self.triggered_timeframes:
            row_idx = self._row_indices_by_tf[timeframe][self._iteration_index]
            for symbol, symbol_data in self._symbol_data_by_tf[timeframe].items():
                first_row = self._stepping_rows.get((timeframe, symbol))
                if first_row is None or row_idx < first_row:
                    continue
                ohlcvs_by_symbol[symbol] = OHLCVRow(
                    timestamp=self._close_indices_by_tf[timeframe][row_idx],
                    open=float(symbol_data["open"][row_idx]),
                    high=float(symbol_data["high"][row_idx]),
                    low=float(symbol_data["low"][row_idx]),
                    close=float(symbol_data["close"][row_idx]),
                    volume=float(symbol_data["volume"][row_idx]),
                )

        return ohlcvs_by_symbol
