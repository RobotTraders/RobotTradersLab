from dataclasses import dataclass
from functools import cached_property

from robottraderslab.exceptions import ProfileConfigError
from robottraderslab.indicators import MAType
from robottraderslab.strategies import Profile
from robottraderslab.strategies.futures import (
    MarginMode,
    SizingRule,
)

DEFAULT_SIGNAL_TYPE = MAType.SMA
DEFAULT_TREND_LENGTH = 200
DEFAULT_TREND_TYPE = MAType.SMA


@dataclass(frozen=True, kw_only=True)
class TrixParameters:
    """The lengths and averages the TRIX signals are built from."""

    trix_length: int
    signal_length: int
    signal_type: MAType = DEFAULT_SIGNAL_TYPE
    trend_length: int = DEFAULT_TREND_LENGTH
    trend_type: MAType = DEFAULT_TREND_TYPE


@dataclass(frozen=True, kw_only=True)
class ProfileConfig(Profile, TrixParameters):
    """One symbol traded on one timeframe, with its own TRIX settings and sizing.

    Profiles may stack on the same symbol, each holding its own position. A
    section failing a check of its own raises `ProfileConfigError`, which
    takes its symbol out of a live run while the other symbols trade. An edit
    of the configuration file that does this runs the bot's last working copy of
    it instead, with that symbol in exit-only mode.
    """

    sizing: SizingRule
    leverage: float = 1.0
    margin_mode: MarginMode = MarginMode.ISOLATED
    long_only: bool = False
    short_only: bool = False
    stop_loss_pct: float | None = None

    def __post_init__(self) -> None:
        """
        Raises:
            ProfileConfigError: If both direction flags are set, or the
                stop-loss is not a fraction of the entry price.
        """
        if self.long_only and self.short_only:
            raise ProfileConfigError(
                "long_only and short_only cannot both be true; leave both out to "
                "trade both directions"
            )
        if self.stop_loss_pct is not None and not 0 < self.stop_loss_pct < 1:
            raise ProfileConfigError(
                f"stop_loss_pct must be in (0, 1), got {self.stop_loss_pct}"
            )

    @cached_property
    def long_entry_column(self) -> str:
        return f"{self.profile_id}_long_entry"

    @cached_property
    def long_exit_column(self) -> str:
        return f"{self.profile_id}_long_exit"

    @cached_property
    def short_entry_column(self) -> str:
        return f"{self.profile_id}_short_entry"

    @cached_property
    def short_exit_column(self) -> str:
        return f"{self.profile_id}_short_exit"
