import logging
from collections.abc import Iterator
from itertools import islice

from robottraderslab._core import Symbol

from .order_models import Order, StopLossOrder, TakeProfitOrder

logger = logging.getLogger(__name__)


class OrderBook:
    """
    A container for pending orders, indexed by symbol.

    The simulation engine controls when orders are removed and when TP/SL
    orders are created. Stop-loss and take-profit guards additionally belong
    to a contingency group held within their symbol's bucket, so resolving
    one member reaches the rest through that symbol alone.
    """

    def __init__(self) -> None:
        self._dead_order_ids: set[int] = set()
        self._orders_by_symbol: dict[Symbol, list[Order]] = {}

    def __iter__(self) -> Iterator[Order]:
        """Iterate over every pending order, skipping any already removed."""
        for bucket in self._orders_by_symbol.values():
            for order in bucket:
                if id(order) not in self._dead_order_ids:
                    yield order

    def add_order(self, order: Order) -> None:
        """Add an order to the book."""
        self._orders_by_symbol.setdefault(order.symbol, []).append(order)
        logger.debug("Order added: %s", order)

    def cancel_order_by_id(self, symbol: Symbol, order_id: str) -> Order | None:
        """Remove the order with this id from the symbol's bucket.

        Returns:
            The order removed, `None` when no pending order matched.
        """
        bucket = self._orders_by_symbol.get(symbol)
        if bucket is None:
            return None
        for index, order in enumerate(bucket):
            if order.order_id == order_id and id(order) not in self._dead_order_ids:
                bucket.pop(index)
                if not bucket:
                    del self._orders_by_symbol[symbol]
                return order
        return None

    def cancel_orders_for_symbol(self, symbol: Symbol) -> list[Order]:
        """Remove every pending order on the symbol.

        Returns:
            The orders removed, so their reservations can be released.
        """
        bucket = self._orders_by_symbol.pop(symbol, [])
        pending = [o for o in bucket if id(o) not in self._dead_order_ids]
        self._dead_order_ids.difference_update(id(o) for o in bucket)
        return pending

    def find_guard(
        self,
        symbol: Symbol,
        order_type: type[StopLossOrder] | type[TakeProfitOrder],
    ) -> StopLossOrder | TakeProfitOrder | None:
        for order in self._orders_by_symbol.get(symbol, ()):
            if id(order) in self._dead_order_ids:
                continue
            if isinstance(order, order_type):
                return order
        return None

    def orders_for(self, symbol: Symbol) -> Iterator[Order]:
        """An order added during iteration, such as a guard a fill opens, is left
        to the next iteration, so a guard is first matched against prices after
        its entry fills.
        """
        bucket = self._orders_by_symbol.get(symbol, [])
        try:
            for order in islice(bucket, len(bucket)):
                if id(order) not in self._dead_order_ids:
                    yield order
        finally:
            self._compact_symbol(symbol)

    def remove_order(self, order: Order) -> None:
        """Remove an order from the book and resolve the rest of its
        contingency group.
        """
        logger.debug("Order removed: %s", order)
        self._dead_order_ids.add(id(order))
        group_id: str | None = getattr(order, "group_id", None)
        self.resolve_group(order.symbol, group_id)

    def resolve_group(self, symbol: Symbol, group_id: str | None) -> None:
        """Cancel every order on the symbol that belongs to the given group.

        A no-op for `None`, so callers can resolve whatever group a position
        happens to carry without checking whether it ever had one.
        """
        if group_id is None:
            return
        for order in self._orders_by_symbol.get(symbol, ()):
            if getattr(order, "group_id", None) == group_id:
                self._dead_order_ids.add(id(order))

    def _compact_symbol(self, symbol: Symbol) -> None:
        bucket = self._orders_by_symbol.get(symbol)
        if bucket is None:
            return
        remaining = [o for o in bucket if id(o) not in self._dead_order_ids]
        self._dead_order_ids.difference_update(id(o) for o in bucket)
        if remaining:
            self._orders_by_symbol[symbol] = remaining
        else:
            del self._orders_by_symbol[symbol]
