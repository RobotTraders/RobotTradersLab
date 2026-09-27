from dataclasses import dataclass
from functools import cached_property
from typing import Annotated

from pydantic import AfterValidator, ConfigDict

from .actions.members import ProfileTag, profile_tag, require_within_budget
from .position_tracker import TrackingId
from .symbol import Symbol
from .timeframes import TimeFrame

_RESERVED_TAG_CHARACTERS = ("@", "-", "/", ":", " ")


def _carryable(tag: str) -> str:
    if any(char in tag for char in _RESERVED_TAG_CHARACTERS):
        raise ValueError(f"'{tag}' cannot contain {list(_RESERVED_TAG_CHARACTERS)}")
    require_within_budget(tag)
    return tag


type Tag = Annotated[str, AfterValidator(_carryable)]


@dataclass(frozen=True, kw_only=True)
class Profile:
    """The market a profile trades and the name it trades under.

    A strategy's profile class extends it with the fields its
    `[[strategy.profiles]]` sections carry, each typed, and the engine builds
    every section into that class by those types. A section holding a key the
    class has no field for is refused. Two profiles on one symbol and
    timeframe are told apart by their `tag`.

    Attributes:
        symbol: The market, parsed from the string the configuration spells.
        timeframe: The candles the profile trades on.
        tag: What tells the profile from another on the same symbol and
            timeframe; empty when nothing has to. At most 8 bytes, without
            `@`, `-`, `/`, `:` or a space, so it fits every venue's client
            order id.
    """

    __pydantic_config__ = ConfigDict(extra="forbid")

    symbol: Symbol
    timeframe: TimeFrame
    tag: Tag = ""

    @cached_property
    def order_tag(self) -> ProfileTag:
        """The tag the profile's orders carry, by which a fill is traced back to it."""
        return profile_tag(self.timeframe, self.tag)

    @cached_property
    def profile_id(self) -> TrackingId:
        """`symbol@timeframe`, followed by `-tag` when a tag is set."""
        if self.tag:
            return TrackingId(f"{self.symbol}@{self.timeframe}-{self.tag}")
        return TrackingId(f"{self.symbol}@{self.timeframe}")
