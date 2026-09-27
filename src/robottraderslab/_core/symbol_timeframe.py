from dataclasses import dataclass

from .symbol import Symbol
from .timeframes import TimeFrame


@dataclass(frozen=True, slots=True)
class SymbolTimeframe:
    """One market a strategy watches: a symbol on a timeframe."""

    symbol: Symbol
    timeframe: TimeFrame
