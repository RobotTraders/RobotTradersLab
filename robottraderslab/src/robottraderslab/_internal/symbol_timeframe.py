from dataclasses import dataclass

from robottraderslab_core import Symbol, TimeFrame


@dataclass(frozen=True, slots=True)
class SymbolTimeframe:
    """One market a strategy watches: a symbol on a timeframe."""

    symbol: Symbol
    timeframe: TimeFrame
