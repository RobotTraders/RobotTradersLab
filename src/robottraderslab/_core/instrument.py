from dataclasses import dataclass


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
