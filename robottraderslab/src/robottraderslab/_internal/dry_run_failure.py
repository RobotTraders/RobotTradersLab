from dataclasses import dataclass

from robottraderslab_core import Symbol
from robottraderslab_core.exceptions import ExchangeRecoverableError


@dataclass(frozen=True, slots=True)
class DryRunFailure:
    """One failure a dry run of a live cycle met.

    Attributes:
        symbol: The symbol the failure belongs to, None when it belongs to no
            single symbol: the strategy's `setup`, the account, a general or
            final slot.
        message: What failed, naming the cause.
        recoverable: Whether the failure is a read the venue or the candles
            did not answer, which proves nothing against the strategy.
    """

    symbol: Symbol | None
    message: str
    recoverable: bool


def failure_of(error: BaseException, symbol: Symbol | None) -> DryRunFailure:
    """The failure an exception a dry run met stands for."""
    return DryRunFailure(
        symbol,
        f"{type(error).__name__}: {error}",
        recoverable=isinstance(error, ExchangeRecoverableError),
    )
