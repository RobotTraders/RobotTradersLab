import math
from typing import TYPE_CHECKING

from robottraderslab._core import (
    AccountSnapshot,
    PositionSide,
    SizingRule,
    Symbol,
    quantity_from_total_balance,
    signed_position,
    validate_price,
)

from .futures_order_builder import FuturesOrderBuilder

if TYPE_CHECKING:
    from .futures_account import FuturesAccount

MINIMUM_ORDER_RATIO = 0.001


class FuturesPositionTarget:
    """Where a position on a symbol should stand, sized with a rule or a
    quantity as an entry is.

    Each sizing returns the order that brings the position there: the
    difference between the target and the position the snapshot reports, in
    one order whichever side of flat the two stand on, reduce-only when it
    only shrinks the position; None when the position already stands within
    the minimum order of the target.
    """

    def __init__(
        self,
        *,
        account: "FuturesAccount",
        symbol: Symbol,
        side: PositionSide,
        account_snapshot: AccountSnapshot,
        minimum_order_ratio: float,
    ) -> None:
        """Initialise the target.

        Args:
            side: Side the position stands on once the order fills.
            minimum_order_ratio: Share of the total balance under which the
                difference is left alone.
        """
        self._account = account
        self._symbol = symbol
        self._side = side
        self._account_snapshot = account_snapshot
        self._minimum_order_ratio = minimum_order_ratio

    def quantity(
        self, quantity: float, current_price: float
    ) -> FuturesOrderBuilder | None:
        """Target a quantity in the base currency.

        Args:
            quantity: What the position holds once the order fills, in the
                base currency; zero to stand flat.
            current_price: Price the minimum order is measured at.

        Raises:
            ValueError: If the quantity is negative or NaN, the price is not
                positive or the symbol carries no margin currency.
        """
        if quantity < 0 or math.isnan(quantity):
            raise ValueError(f"`quantity` must be zero or above; received {quantity}")
        validate_price("current_price", current_price)
        return self._order_to(quantity, current_price)

    def size(self, rule: SizingRule, price: float) -> FuturesOrderBuilder | None:
        """Target the position a rule sizes, typically the one the
        configuration names for the profile.

        Args:
            price: Price the rule converts its amount into a quantity at, and
                the minimum order is measured at.

        Raises:
            StrategyCriticalError: If the rule cannot size the position, as a
                risk rule cannot without a stop-loss.
            ValueError: If the price is not positive or the symbol carries no
                margin currency.
        """
        validate_price("price", price)
        return self._order_to(
            rule.quantity(
                self._symbol,
                price,
                self._account_snapshot,
                placement_reserve_rate=self._account.placement_reserve_rate,
                placement_requirement_rate=self._account.placement_requirement_rate,
                stop_loss_price=None,
            ),
            price,
        )

    def _order_to(
        self, target_quantity: float, current_price: float
    ) -> FuturesOrderBuilder | None:
        target = (
            target_quantity if self._side is PositionSide.LONG else -target_quantity
        )
        position = self._account_snapshot.position(self._symbol)
        held = 0.0 if position is None else signed_position(position)
        difference = target - held
        minimum = quantity_from_total_balance(
            self._minimum_order_ratio,
            current_price,
            self._account_snapshot,
            self._symbol,
            0.0,
        )
        if abs(difference) < minimum:
            return None
        reduce_only = _shrinks_without_crossing(held, target)
        if difference > 0:
            return self._account._create_buy_order(
                self._symbol, difference, reduce_only=reduce_only
            )
        return self._account._create_sell_order(
            self._symbol, -difference, reduce_only=reduce_only
        )


def _shrinks_without_crossing(held: float, target: float) -> bool:
    """True only when the order narrows the position towards zero without
    opening the other side.
    """
    if held == 0.0 or abs(target) >= abs(held):
        return False
    return target == 0.0 or math.copysign(1.0, target) == math.copysign(1.0, held)
