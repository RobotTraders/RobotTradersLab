from dataclasses import dataclass
from enum import StrEnum


class TradingMode(StrEnum):
    """Where the candles a strategy trades on come from.

    In backtest the candles are replayed from history, one after the other,
    and every fill is simulated. In live they arrive as the market prints
    them, one per run, and orders reach the venue.
    """

    BACKTEST = "backtest"
    LIVE = "live"


@dataclass(frozen=True, slots=True)
class TradingSystem:
    """The run a strategy is part of, and the mode it is trading in."""

    trading_mode: TradingMode
