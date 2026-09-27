from datetime import datetime
from typing import Protocol

import pandas as pd

from robottraderslab._core import Symbol, TimeFrame


class OhlcvRepositoryProtocol(Protocol):
    """
    Protocol for OHLCV data persistence.

    Time semantics and invariants:
    - All inputs and outputs at the repository boundary use UTC-aware
      `pandas.DateTimeIndex` sorted in ascending order.
    - Stored DataFrames must already be normalized using
      `ohlcv_provider.date_utils.standardize_ohlcv_index`.
    - Callers pass `start_date`/`end_date` as timezone-aware UTC datetimes; the
      repository applies inclusive slicing on these bounds.
    """

    async def store(
        self, symbol: Symbol, timeframe: TimeFrame, data: pd.DataFrame
    ) -> None: ...

    async def load(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame: ...

    async def exists(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> bool: ...

    async def delete(self, symbol: Symbol, timeframe: TimeFrame) -> None: ...
