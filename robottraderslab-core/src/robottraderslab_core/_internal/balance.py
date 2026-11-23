from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True, kw_only=True, slots=True)
class Balance:
    """Funds held in one currency, in total and net of what is locked."""

    locked: float
    total: float

    @property
    def available(self) -> float:
        """Funds free to use: the total less what is locked."""
        return self.total - self.locked

    def update_copy(
        self,
        *,
        locked_delta: float = 0.0,
        total_delta: float = 0.0,
    ) -> "Balance":
        """Return a copy with the deltas applied.

        Args:
            locked_delta: Amount to add to locked, negative to release.
            total_delta: Amount to add to total, negative to spend.

        Raises:
            RuntimeError: If the update would lock more than is available.
        """
        if locked_delta > self.available:
            raise RuntimeError(
                f"Insufficient available funds: {self.available} available, "
                f"operation requires {locked_delta}"
            )

        return Balance(
            locked=max(0.0, self.locked + locked_delta),
            total=max(0.0, self.total + total_delta),
        )

    @classmethod
    def compute(
        cls,
        *,
        available: float | None = None,
        locked: float | None = None,
        total: float | None = None,
    ) -> Self:
        """Build a balance from any two of the three amounts.

        Args:
            available: Funds free to use; computed when omitted.
            locked: Funds backing orders and positions; computed when omitted.
            total: Sum of the other two; computed when omitted.

        Raises:
            RuntimeError: If fewer than two amounts are provided.
        """
        try:
            total = float(available + locked if total is None else total)  # type: ignore[operator]
            locked = float(total - available if locked is None else locked)  # type: ignore[operator]
        except TypeError as e:
            raise RuntimeError(
                "Provide at least two of `available`, `locked`, and `total`. The other one is computed."
            ) from e

        return cls(locked=locked, total=total)
