import math
from dataclasses import dataclass
from typing import Self

from robottraderslab.strategies import OHLCVs, PositionTracker, TrackedPosition

from .profile_config import ProfileConfig


@dataclass(frozen=True, slots=True)
class ProfileSnapshot:
    """Per-profile view of the current candle: its signals plus tracked state."""

    price: float
    long_entry: bool
    long_exit: bool
    short_entry: bool
    short_exit: bool
    position: TrackedPosition | None

    @classmethod
    def build(
        cls,
        profile: ProfileConfig,
        ohlcvs: OHLCVs,
        position_tracker: PositionTracker,
    ) -> Self:
        """Read one profile's signals and tracked position for the current candle."""
        symbol = profile.symbol
        timeframe = profile.timeframe
        return cls(
            price=ohlcvs.current(symbol, timeframe, "close"),
            long_entry=ohlcvs.signal(symbol, timeframe, profile.long_entry_column),
            long_exit=ohlcvs.signal(symbol, timeframe, profile.long_exit_column),
            short_entry=ohlcvs.signal(symbol, timeframe, profile.short_entry_column),
            short_exit=ohlcvs.signal(symbol, timeframe, profile.short_exit_column),
            position=position_tracker.get(profile.profile_id),
        )

    @property
    def has_no_price(self) -> bool:
        return math.isnan(self.price)
