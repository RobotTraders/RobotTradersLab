from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SimulatedMarket:
    """A simulated exchange paired with the account that trades against it."""

    simulator: Any
    account: Any
