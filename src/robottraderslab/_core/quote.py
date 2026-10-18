from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True, slots=True)
class Quote:
    bid: float
    ask: float
