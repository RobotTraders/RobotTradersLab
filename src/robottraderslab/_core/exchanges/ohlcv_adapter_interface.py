from types import TracebackType
from typing import Any, Protocol, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..symbol import Symbol
from ..timeframes import TimeFrame

type OhlcvData = npt.NDArray[np.float64]

_OHLCV_COLUMNS = 6


def no_candles() -> OhlcvData:
    """Answer of a source with no candles to give for a range.

    Shaped like every other answer, so no caller has to tell emptiness apart.
    """
    return np.empty((0, _OHLCV_COLUMNS), dtype=np.float64)


class OhlcvAdapterProtocol(Protocol):
    """A common interface over OHLCV sources whose own APIs differ.

    Range semantics:
    - Adapters are asked for a range and answer with the whole of it, so how
      many requests it takes, which endpoint serves which part of it and how
      many candles a request may carry stay inside the adapter.

    Candle semantics:
    - Candles travel as one array of shape (rows, 6) and dtype float64, its
      columns being timestamp, open, high, low, close and volume in that order.
    - The array is the only representation the whole chain speaks, from the
      response an adapter parses to the dataframe a consumer builds.

    Time semantics and invariants:
    - All time-related values are interpreted as UTC.
    - Input bounds that use epoch time must represent UTC milliseconds since Unix epoch.
    - Returned timestamps are UTC milliseconds since Unix epoch.
    - Consumers must normalize returned dataframes/series using `ohlcv_provider.date_utils`
      (e.g., `standardize_ohlcv_index_from_timestamp`) at the boundary.

    Validation semantics:
    - Implementations must validate symbol and timeframe support internally.
    - Invalid symbols should raise InvalidSymbolError.
    - Invalid timeframes should raise InvalidTimeframeError.

    Factory pattern:
    - A shared `.create(**kwargs)` factory keeps instantiation consistent
      across every adapter.
    """

    @classmethod
    def dataset_qualifiers(cls, **kwargs: Any) -> tuple[str, ...]:
        """Name each configured choice that changes the candles themselves.

        One source can serve the same symbol and timeframe as several series
        whose numbers differ, and a candle from one cannot stand in for a
        candle from another. Each series is held under its own name, and
        these qualifiers are what give it one.

        A choice qualifies when it changes the numbers.

        Args:
            **kwargs: The adapter configuration, as `create` receives it.

        Returns:
            A qualifier per choice that departs from what this source serves
            by default, empty when every choice is the default one. Order does
            not matter.
        """
        return ()

    @classmethod
    def create(cls, **kwargs: Any) -> Self:
        """Factory method to create an instance of the adapter."""
        ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

    async def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_ms: int,
        end_ms: int,
        include_open_candle: bool = False,
    ) -> OhlcvData:
        """Fetch the candles of a whole range, in as many requests as it takes.

        Args:
            start_ms: Start of the range, in UTC milliseconds since the epoch.
            end_ms: End of the range, in UTC milliseconds since the epoch.
            include_open_candle: Keep the candle whose period has not elapsed
                yet, for a caller that wants the market as it stands. Sources
                that never report it answer the same either way.

        Returns:
            Candles of the range, the one currently forming left out unless
            it was asked for. Rows may repeat or arrive unordered; callers sort
            and deduplicate them.
        """
        ...

    @classmethod
    def market_open_mask(cls, index: pd.DatetimeIndex) -> np.ndarray | None:
        """Mark which timestamps fall within market trading hours.

        The calendar belongs to the venue, so it is read from the adapter's
        class before any adapter is open. Lets consumers distinguish genuine
        data gaps from scheduled market closures.

        Returns:
            Boolean mask aligned with the index, True where the market is
            open, or None when the market never closes.
        """
        return None
