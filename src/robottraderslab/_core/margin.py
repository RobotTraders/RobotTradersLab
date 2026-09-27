from dataclasses import dataclass
from enum import StrEnum


class MarginMode(StrEnum):
    """How margin backs a position.

    Isolated confines the loss to the margin posted for that position, cross
    draws on the whole balance.
    """

    ISOLATED = "isolated"
    CROSS = "cross"


@dataclass(kw_only=True, frozen=True, slots=True)
class MarginSettings:
    """The margin mode and leverage of one symbol: what the venue holds when
    the snapshot reports them, what the engine keeps when `margin_targets`
    declares them.

    Attributes:
        leverage: None when the venue defines no per-symbol leverage in the
            margin mode, and in a target, to leave the leverage as the venue
            holds it.
    """

    leverage: float | None
    margin_mode: MarginMode
