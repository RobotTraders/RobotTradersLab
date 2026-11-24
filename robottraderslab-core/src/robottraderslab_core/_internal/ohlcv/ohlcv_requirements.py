from dataclasses import dataclass

from ..symbol import Symbol
from ..timeframes import TimeFrame


@dataclass(frozen=True, slots=True)
class OHLCVRequirement:
    """Single OHLCV data requirement."""

    symbol: Symbol
    timeframe: TimeFrame
    lookback: int = 0


class OHLCVRequirements:
    """Declares the OHLCV data a strategy needs."""

    def __init__(self) -> None:
        self._requirements: list[OHLCVRequirement] = []

    def add(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        lookback: int = 0,
    ) -> None:
        """Declare a symbol the strategy trades, on one timeframe.

        Args:
            lookback: Candles loaded before the trading window, the warm-up
                history the strategy's indicators need.
        """
        self._requirements.append(OHLCVRequirement(symbol, timeframe, lookback))

    def _get_all(self) -> list[OHLCVRequirement]:
        return list(self._requirements)
