from collections.abc import Callable
from typing import Protocol

import pandas as pd

from ..symbol import Symbol
from ..timeframes import TimeFrame

type OHLCVFetcher = Callable[[Symbol, TimeFrame], pd.DataFrame]


class OHLCVProviderProtocol(Protocol):
    def set_dates(self, start_date: str, end_date: str) -> None: ...

    def fetch_ohlcv(self, symbol: Symbol, timeframe: TimeFrame) -> pd.DataFrame: ...

    def get_all_cached_ohlcv_for_symbol(self, symbol: Symbol) -> pd.DataFrame: ...
