import logging
import math
from collections.abc import Generator, Iterable
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..candle_grid import candle_step
from ..exceptions import MissingOhlcvDataError, StrategyCriticalError
from ..symbol import Symbol
from ..timeframes import TimeFrame, to_seconds
from .timeframe_snapshot import TimeframeSnapshot, _SymbolData

_logger = logging.getLogger(__name__)

OHLCVsByTimeframeAndSymbol = dict[TimeFrame, dict[Symbol, pd.DataFrame]]


def _as_utc64(timestamp: datetime) -> np.datetime64:
    """Read a timestamp as UTC, which is how the providers write candle times."""
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(UTC).replace(tzinfo=None)
    return np.datetime64(timestamp)


class OHLCVs:
    """The candles a strategy declared, and the columns it computes on them.

    `column` and `add_column` work on a whole series; `current` and `signal`
    read the candle the strategy is standing on, which the engine advances.
    The engine stands on the moments candles close, so on a timeframe with no
    close at this moment, `current` and `signal` read its last closed candle.
    A read on a pair the run's candles do not hold, never declared or dropped
    from a live cycle after a failed fetch, raises `MissingOhlcvDataError`.
    """

    def __init__(self, ohlcvs: OHLCVsByTimeframeAndSymbol) -> None:
        self._store = _OHLCVStore(ohlcvs)
        self._current_iteration_index: int = 0

    def column(
        self, symbol: Symbol, timeframe: TimeFrame, name: str
    ) -> npt.NDArray[Any]:
        """Return a full column, one entry per candle.

        The dtype is the column's own: market data is float, and a column a
        strategy wrote comes back as it was written, so a boolean signal
        stays boolean.

        Args:
            name: Column name (e.g. "close", "open", "volume").

        Raises:
            KeyError: If no column carries the name.
        """
        return self._store.column(symbol, timeframe, name)

    def covers(self, symbol: Symbol, timeframe: TimeFrame, moment: datetime) -> bool:
        """Report whether a moment falls between the first loaded candle's
        open and the last one's close, both included.
        """
        return self._store.covers(symbol, timeframe, moment)

    def candles_between(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        up_to: datetime,
        after: datetime | None = None,
    ) -> slice:
        """Return the slice of a column holding the candles lying whole inside
        a stretch of time.

        Args:
            up_to: The candles closed at or before it are included.
            after: The candles opened at or after it are included; from the
                first candle when omitted.
        """
        return self._store.candles_between(symbol, timeframe, up_to, after)

    def timestamps(
        self, symbol: Symbol, timeframe: TimeFrame
    ) -> npt.NDArray[np.datetime64]:
        """Return the moment every candle closed, in UTC, one entry per candle.

        The entry at an index is when the candle at that index in `column`
        closed, which is when its values became known.
        """
        return self._store.close_times(symbol, timeframe)

    def add_column(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        name: str,
        values: npt.ArrayLike,
    ) -> None:
        """Store a column the strategy computed on the pair's candles.

        Args:
            name: Column name (e.g. "buy", "fast_sma").
            values: One entry per candle, in candle order.

        Raises:
            ValueError: If the values are not one per candle.
        """
        self._store.add_column(symbol, timeframe, name, values)

    def current(self, symbol: Symbol, timeframe: TimeFrame, name: str) -> float:
        """Read a column's value on the candle the strategy is standing on.

        Args:
            name: Column name (e.g. "close", "fast_sma").

        Raises:
            KeyError: If no column carries the name.
            MissingOhlcvDataError: If none of the timeframe's candles had
                closed by the moment.
        """
        return self._store.value_at(
            timeframe, symbol, name, self._current_iteration_index
        )

    def signal(self, symbol: Symbol, timeframe: TimeFrame, name: str) -> bool:
        """Report whether a column is raised on the candle the strategy is standing on.

        A NaN, such as an indicator's warm-up, is not raised.

        Args:
            name: Column name (e.g. "buy", "long_entry").

        Raises:
            KeyError: If no column carries the name.
            MissingOhlcvDataError: If none of the timeframe's candles had
                closed by the moment.
        """
        value = self._store.value_at(
            timeframe, symbol, name, self._current_iteration_index
        )
        return value != 0 and not math.isnan(value)

    def _iter_timeframes(
        self,
        start_idx: int = 0,
        after: datetime | None = None,
    ) -> Generator[TimeframeSnapshot, None, None]:
        """Uses the union of all timeframes' closes so that early-available
        timeframes are not blocked by late-arriving ones.

        Args:
            start_idx: Index offset into the moments (e.g. -1 for the last).
            after: If given, only the moments strictly after it, so a run
                starting on a candle's open decides first on that candle's
                close. Takes precedence over start_idx.
        """
        timestamps = self._store.iteration_timestamps()
        if after is not None:
            timestamps = timestamps[timestamps > pd.Timestamp(after)]
        else:
            timestamps = timestamps[start_idx:]

        self._store.precompute(timestamps)
        close_indices_by_tf = self._store.close_indices_by_tf()
        tf_map = self._store.get_timestamp_to_timeframes_map()

        for i, timestamp in enumerate(timestamps):
            self._current_iteration_index = i
            yield TimeframeSnapshot(
                timestamp,
                tf_map[timestamp],
                self._store._symbol_data_by_tf,
                close_indices_by_tf,
                self._store._row_indices_by_tf,
                i,
                self._store._stepping_rows,
            )

    def _has_timeframe(self, timeframe: TimeFrame) -> bool:
        return timeframe in self._store._dataframes

    def _last_candle_open(self, timeframe: TimeFrame) -> datetime:
        index = next(iter(self._store._dataframes[timeframe].values())).index
        return cast(datetime, index[-1])

    def _last_candle_close(self, timeframe: TimeFrame) -> datetime:
        return cast(datetime, self._store.close_index(timeframe)[-1])

    def _earliest_candle_open(self) -> datetime:
        """Return the open of the first candle of the loaded window.

        Raises:
            StrategyCriticalError: If no series holds a single candle.
        """
        starts = [
            symbol_df.index[0]
            for symbol_dict in self._store._dataframes.values()
            for symbol_df in symbol_dict.values()
            if not symbol_df.empty
        ]
        if not starts:
            raise StrategyCriticalError(
                "No candles were loaded for any declared symbol and timeframe."
            )
        return cast(datetime, min(starts))

    def _warn_if_data_starts_late(self, start_after: datetime) -> None:
        """Log one warning per symbol/timeframe when exchange data starts late."""
        for timeframe, symbol_dict in self._store._dataframes.items():
            for symbol, symbol_df in symbol_dict.items():
                close = symbol_df[symbol_df.index >= start_after]["close"]
                first_valid = cast(pd.Timestamp | None, close.first_valid_index())
                if first_valid is None:
                    _logger.warning(
                        "%s@%s: no data on exchange, skipped in backtest",
                        symbol,
                        timeframe,
                    )
                elif first_valid > start_after:
                    _logger.warning(
                        "%s@%s: no data on exchange before %s, backtest starts from there",
                        symbol,
                        timeframe,
                        first_valid.date(),
                    )


class _OHLCVStore:
    """Columnar data store encapsulating OHLCV storage and precomputation."""

    def __init__(self, ohlcvs: OHLCVsByTimeframeAndSymbol) -> None:
        """Initialize store with raw OHLCV DataFrames.

        Args:
            ohlcvs: Copied, so that `add_column` does not leak
                strategy-computed columns back to the caller.
        """
        self._dataframes: OHLCVsByTimeframeAndSymbol = {
            tf: {symbol: df.copy() for symbol, df in symbol_dict.items()}
            for tf, symbol_dict in ohlcvs.items()
        }
        self._symbol_data_by_tf: dict[TimeFrame, dict[Symbol, _SymbolData]] = {}
        self._columns: dict[tuple[TimeFrame, Symbol, str], npt.NDArray[Any]] = {}
        self._open_times: dict[
            tuple[TimeFrame, Symbol], npt.NDArray[np.datetime64]
        ] = {}
        self._close_times: dict[
            tuple[TimeFrame, Symbol], npt.NDArray[np.datetime64]
        ] = {}
        self._close_indices: dict[TimeFrame, pd.DatetimeIndex] = {}
        self._row_indices_by_tf: dict[TimeFrame, npt.NDArray[np.intp]] = {}
        self._stepping_rows: dict[tuple[TimeFrame, Symbol], int] = {}
        self._timestamp_to_timeframes_map: dict[pd.Timestamp, list[TimeFrame]] = {}

    def column(
        self, symbol: Symbol, timeframe: TimeFrame, name: str
    ) -> npt.NDArray[Any]:
        """Return a full column, extracted once and kept for the next reader."""
        key = (timeframe, symbol, name)
        cached = self._columns.get(key)
        if cached is None:
            frame = self._frame(symbol, timeframe)
            try:
                series = frame[name]
            except KeyError:
                raise _unknown_column_error(
                    symbol, timeframe, name, frame.columns
                ) from None
            cached = series.to_numpy()
            self._columns[key] = cached
        return cached

    def covers(self, symbol: Symbol, timeframe: TimeFrame, moment: datetime) -> bool:
        opens = self._open_times_of(symbol, timeframe)
        if opens.size == 0:
            return False
        wanted = _as_utc64(moment)
        closes = self.close_times(symbol, timeframe)
        return bool(opens[0] <= wanted <= closes[-1])

    def _frame(self, symbol: Symbol, timeframe: TimeFrame) -> pd.DataFrame:
        """Return the pair's frame.

        Raises:
            MissingOhlcvDataError: If the pair was not loaded: never declared,
                or dropped from a live cycle after its fetch failed.
        """
        try:
            return self._dataframes[timeframe][symbol]
        except KeyError:
            raise MissingOhlcvDataError(symbol, timeframe) from None

    def candles_between(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        up_to: datetime,
        after: datetime | None,
    ) -> slice:
        start = (
            0
            if after is None
            else int(
                self._open_times_of(symbol, timeframe).searchsorted(
                    _as_utc64(after), side="left"
                )
            )
        )
        stop = int(
            self.close_times(symbol, timeframe).searchsorted(
                _as_utc64(up_to), side="right"
            )
        )
        return slice(start, stop)

    def close_times(
        self, symbol: Symbol, timeframe: TimeFrame
    ) -> npt.NDArray[np.datetime64]:
        cached = self._close_times.get((timeframe, symbol))
        if cached is None:
            opens = pd.DatetimeIndex(self._open_times_of(symbol, timeframe))
            cached = (opens + candle_step(timeframe)).to_numpy()
            self._close_times[(timeframe, symbol)] = cached
        return cached

    def _open_times_of(
        self, symbol: Symbol, timeframe: TimeFrame
    ) -> npt.NDArray[np.datetime64]:
        cached = self._open_times.get((timeframe, symbol))
        if cached is None:
            index = pd.DatetimeIndex(self._frame(symbol, timeframe).index)
            if index.tz is not None:
                index = index.tz_convert(UTC).tz_localize(None)
            cached = index.to_numpy()
            self._open_times[(timeframe, symbol)] = cached
        return cached

    def add_column(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        name: str,
        values: npt.ArrayLike,
    ) -> None:
        """Register a strategy-computed column for a symbol/timeframe pair."""
        frame = self._frame(symbol, timeframe)
        column = np.asarray(values)
        if column.shape != (len(frame),):
            raise ValueError(
                f"{symbol}@{timeframe}: column '{name}' was given "
                f"{_describe_values(column)} for {len(frame)} candles"
            )
        frame[name] = column
        self._columns.pop((timeframe, symbol, name), None)

    def value_at(
        self,
        timeframe: TimeFrame,
        symbol: Symbol,
        name: str,
        iteration_index: int,
    ) -> float:
        """Look up a single scalar value at a precomputed row index.

        Raises:
            MissingOhlcvDataError: If the pair was not loaded, or none of its
                candles had closed by the iteration.
        """
        try:
            row_idx = self._row_indices_by_tf[timeframe][iteration_index]
            data = self._symbol_data_by_tf[timeframe][symbol]
        except KeyError:
            raise MissingOhlcvDataError(symbol, timeframe) from None
        if row_idx < 0:
            raise MissingOhlcvDataError(
                symbol, timeframe, detail="no candle had closed yet"
            )
        try:
            column = data[name]
        except KeyError:
            raise _unknown_column_error(symbol, timeframe, name, data) from None
        return float(column[row_idx])

    def precompute(self, timestamps: pd.DatetimeIndex) -> None:
        """Build row indices and merge all columns for fast iteration."""
        self._row_indices_by_tf = self._precompute_row_indices(timestamps)
        self._symbol_data_by_tf = self._merge_all_columns()
        self._stepping_rows = self._precompute_stepping_rows()

    def iteration_timestamps(self) -> pd.DatetimeIndex:
        indices = [self.close_index(timeframe) for timeframe in self._dataframes]
        combined: pd.DatetimeIndex = indices[0]
        for idx in indices[1:]:
            combined = pd.DatetimeIndex(combined.union(idx))
        return pd.DatetimeIndex(combined.sort_values())

    def get_timestamp_to_timeframes_map(
        self,
    ) -> dict[pd.Timestamp, list[TimeFrame]]:
        """Return cached mapping of timestamps to triggered timeframes."""
        if not self._timestamp_to_timeframes_map:
            self._compute_timestamp_to_timeframes_map()
        return self._timestamp_to_timeframes_map

    def close_index(self, timeframe: TimeFrame) -> pd.DatetimeIndex:
        """Every symbol of a timeframe shares one candle grid, so the first
        symbol's closes stand for all of them.
        """
        cached = self._close_indices.get(timeframe)
        if cached is None:
            opens = pd.DatetimeIndex(
                next(iter(self._dataframes[timeframe].values())).index
            )
            cached = pd.DatetimeIndex(opens + candle_step(timeframe))
            self._close_indices[timeframe] = cached
        return cached

    def close_indices_by_tf(self) -> dict[TimeFrame, pd.DatetimeIndex]:
        return {
            timeframe: self.close_index(timeframe) for timeframe in self._dataframes
        }

    def _precompute_row_indices(
        self, timestamps: pd.DatetimeIndex
    ) -> dict[TimeFrame, npt.NDArray[np.intp]]:
        """Point each iteration at every timeframe's last candle closed by
        then, -1 before its first close.
        """
        return {
            timeframe: self.close_index(timeframe).searchsorted(
                timestamps, side="right"
            )
            - 1
            for timeframe in self._dataframes
        }

    def _merge_all_columns(self) -> dict[TimeFrame, dict[Symbol, _SymbolData]]:
        return {
            tf: {
                symbol: {col: df[col].to_numpy() for col in df.columns}
                for symbol, df in symbol_dict.items()
            }
            for tf, symbol_dict in self._dataframes.items()
        }

    def _compute_timestamp_to_timeframes_map(self) -> None:
        """A timeframe triggers at the moments one of its candles closes,
        listed shortest first.
        """
        all_timestamps = self.iteration_timestamps()
        timestamp_map: dict[pd.Timestamp, list[TimeFrame]] = {
            timestamp: [] for timestamp in all_timestamps
        }
        for timeframe in sorted(self._dataframes, key=to_seconds):
            common_timestamps = all_timestamps.intersection(self.close_index(timeframe))
            for ts in common_timestamps:
                timestamp_map[ts].append(timeframe)
        self._timestamp_to_timeframes_map = timestamp_map

    def _precompute_stepping_rows(
        self,
    ) -> dict[tuple[TimeFrame, Symbol], int]:
        """Pair each symbol with its shortest timeframe, whose candles alone
        step it in the simulator, and with that series' first valid row: a
        longer candle spans shorter ones already simulated, and at the end of
        a window it reaches past the shorter series.
        """
        shortest: dict[Symbol, TimeFrame] = {}
        for tf in sorted(self._symbol_data_by_tf, key=to_seconds, reverse=True):
            for symbol in self._symbol_data_by_tf[tf]:
                shortest[symbol] = tf
        return {
            (tf, symbol): int(
                np.argmax(~np.isnan(self._symbol_data_by_tf[tf][symbol]["close"]))
            )
            for symbol, tf in shortest.items()
        }


def _unknown_column_error(
    symbol: Symbol, timeframe: TimeFrame, name: str, columns: Iterable[str]
) -> KeyError:
    return KeyError(
        f"{symbol}@{timeframe} has no column '{name}'; its columns are "
        f"{', '.join(columns)}"
    )


def _describe_values(column: npt.NDArray[Any]) -> str:
    if column.ndim == 1:
        return f"{column.size} values"
    return f"values of shape {column.shape}"
