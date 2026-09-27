from datetime import datetime

import pandas as pd

from robottraderslab._core import Symbol, TimeFrame


class NullOhlcvRepository:
    """No-op OHLCV repository for live bots that don't need disk persistence."""

    async def store(
        self, symbol: Symbol, timeframe: TimeFrame, data: pd.DataFrame
    ) -> None:
        pass

    async def load(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    async def exists(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> bool:
        return False

    async def delete(self, symbol: Symbol, timeframe: TimeFrame) -> None:
        pass
