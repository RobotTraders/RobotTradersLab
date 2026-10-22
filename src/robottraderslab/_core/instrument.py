import asyncio
from dataclasses import dataclass, field
from decimal import Decimal

from .account_requirements import AccountRequirements
from .exceptions import ExchangeRecoverableError
from .symbol import Symbol


@dataclass(frozen=True, kw_only=True, slots=True)
class Instrument:
    """The trading rules a venue enforces on a symbol.

    Attributes:
        quantity_step: Quantity increment an order is a whole multiple of.
        minimum_order_value: Smallest order value, in the symbol's quote
            currency, the venue accepts; `None` where none is stated.
        maximum_leverage: `None` where the venue states none.
        maximum_order_quantity: Largest quantity one limit order may carry;
            `None` where the venue states none.
        maximum_market_order_quantity: Largest quantity one market order may
            carry; `None` where the venue states none.
    """

    quantity_step: float
    minimum_order_value: float | None
    maximum_leverage: float | None
    maximum_order_quantity: float | None
    maximum_market_order_quantity: float | None
    _step_numerator: int = field(init=False, repr=False, compare=False)
    _step_denominator: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        numerator, denominator = Decimal(repr(self.quantity_step)).as_integer_ratio()
        object.__setattr__(self, "_step_numerator", numerator)
        object.__setattr__(self, "_step_denominator", denominator)

    def order_quantity(
        self,
        symbol: Symbol,
        quantity: float,
        *,
        price: float | None,
        reduce_only: bool,
    ) -> float:
        """The quantity is cut down to a whole number of steps and never up,
        and comes back as the exact decimal those steps spell, so a venue
        formatting it to its own precision drops none.

        A reduce-only order is held to the step alone, since it is the order
        that closes a position the market has moved under the minimum order
        value.

        Args:
            price: The price the order's value is measured at for the minimum
                order value.

        Raises:
            ExchangeRecoverableError: If the quantity is under one step, or an
                order that may grow the position is worth less than the
                minimum order value at the price.
        """
        numerator = self._step_numerator
        denominator = self._step_denominator
        steps = round(quantity * denominator / numerator)
        tradable = steps * numerator / denominator
        if tradable > quantity:
            steps -= 1
            tradable = steps * numerator / denominator
        if steps <= 0:
            raise ExchangeRecoverableError(
                f"Order quantity {quantity} of {symbol} is under one quantity "
                f"step of {self.quantity_step:g}"
            )
        if (
            not reduce_only
            and price is not None
            and self.minimum_order_value is not None
            and tradable * price < self.minimum_order_value
        ):
            raise ExchangeRecoverableError(
                f"Order of {tradable:g} {symbol} at {price:g} is worth "
                f"{tradable * price:g}, under the minimum order value of "
                f"{self.minimum_order_value:g}"
            )
        return tradable


async def read_instruments(requirements: AccountRequirements) -> None:
    """Run once the strategy has declared its accounts, before it books.

    Each account then holds the trading rules of every symbol it declared, and
    every order it builds is held to them.
    """
    await asyncio.gather(
        *(
            requirement.account._read_instruments(requirement.symbols)
            for requirement in requirements._get_all()
        )
    )
