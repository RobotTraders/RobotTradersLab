from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from .symbol import Symbol


class TradingStatus(StrEnum):
    """What a venue that closes a symbol still lets a bot send on it."""

    CLOSED_TO_NEW_POSITIONS = "closed_to_new_positions"
    CLOSED_TO_EVERY_ORDER = "closed_to_every_order"


@dataclass(frozen=True, slots=True)
class Closure:
    """How a venue closes a symbol.

    Attributes:
        status: What the venue still lets a bot send on the symbol.
        venue_status: The venue's own word for the closure.
    """

    status: TradingStatus
    venue_status: str

    @classmethod
    def from_words(
        cls, statuses: Mapping[str, TradingStatus], words: Iterable[str]
    ) -> Self | None:
        """Return the closure of the first word the statuses name.

        Args:
            statuses: The status each of the venue's closing words stands for.
            words: The words the venue gave the symbol, in the order they
                count.

        Returns:
            `None` where no word closes the symbol, which leaves it open.
        """
        for word in words:
            status = statuses.get(word)
            if status is not None:
                return cls(status, word)
        return None


@dataclass(frozen=True, slots=True)
class Halt:
    """A symbol an account's venue closed.

    Attributes:
        symbol: The halted symbol.
        venue: The exchange the account trades on, as the configuration
            names it.
        account: The name of the account that trades the symbol.
        closure: How the venue closes the symbol.
    """

    symbol: Symbol
    venue: str
    account: str
    closure: Closure
