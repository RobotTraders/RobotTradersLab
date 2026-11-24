from dataclasses import dataclass, field
from itertools import count
from typing import Any

from robottraderslab._internal import trailing_level
from robottraderslab_core import Symbol
from robottraderslab_core.exchanges import (
    OrderSide,
    OrderType,
    PositionSide,
    StopLoss,
    TakeProfit,
    TimeInForce,
    TriggerType,
)

_order_ids = count(1)
_group_ids = count(1)
_execution_ids = count(1)


def new_order_id() -> str:
    """Return a fresh order id, drawn from a process-wide counter.

    A liquidation has no placed order behind it, so its execution still needs
    one to report.
    """
    return str(next(_order_ids))


def _get_order_id_field() -> Any:
    return field(default_factory=new_order_id, init=False)


def new_group_id() -> str:
    """Return a fresh contingency-group id, drawn from a process-wide counter."""
    return f"g{next(_group_ids)}"


def new_execution_id() -> str:
    """One order can fill in several executions, so the id is its own sequence,
    distinct from the order id a fill's pieces share.
    """
    return f"x{next(_execution_ids)}"


@dataclass(kw_only=True, slots=True)
class _Order:
    order_id: str = _get_order_id_field()
    quantity: float
    side: OrderSide
    symbol: Symbol
    kind: OrderType
    trigger_price: float | None = None
    reduce_only: bool = False
    stop_loss: StopLoss | None = None
    take_profit: TakeProfit | None = None
    reason: str | None = None
    limit_price: float | None = None
    client_order_id: str | None = None
    extra_fields: dict[str, Any] | None = None
    fired_level: float | None = None

    @property
    def remaining_quantity(self) -> float:
        return self.quantity

    @property
    def stop_loss_price(self) -> float | None:
        return self.stop_loss.trigger_price if self.stop_loss else None

    @property
    def stop_loss_trigger_type(self) -> TriggerType | None:
        return self.stop_loss.trigger_type if self.stop_loss else None

    @property
    def take_profit_price(self) -> float | None:
        return self.take_profit.trigger_price if self.take_profit else None

    @property
    def take_profit_trigger_type(self) -> TriggerType | None:
        return self.take_profit.trigger_type if self.take_profit else None

    @property
    def trailing_distance(self) -> None:
        return None

    @property
    def trigger_type(self) -> None:
        return None


@dataclass(kw_only=True, slots=True)
class LimitOrder(_Order):
    """A limit order is settled against the close of the candle it is placed
    on once, when its time in force says it cannot wait; `settled` records
    that this happened.
    """

    kind: OrderType = OrderType.LIMIT
    limit_price: float
    time_in_force: TimeInForce = TimeInForce.GTC
    settled: bool = False
    locked_margin: float = 0.0

    def __str__(self) -> str:
        return f"limit order {self.side} {self.quantity:,.6f} of {self.symbol} at {self.limit_price}"


@dataclass(kw_only=True, slots=True)
class MarketOrder(_Order):
    kind: OrderType = OrderType.MARKET

    @property
    def time_in_force(self) -> None:
        return None

    def __str__(self) -> str:
        return f"market order {self.side} {self.quantity:,.6f} of {self.symbol}"


class _RestingProtection:
    """A protection rests whole at its trigger until it fires."""

    __slots__ = ()
    quantity: float | None

    @property
    def reduce_only(self) -> bool:
        return True

    @property
    def remaining_quantity(self) -> float | None:
        return self.quantity

    @property
    def limit_price(self) -> None:
        return None

    @property
    def stop_loss_price(self) -> None:
        return None

    @property
    def stop_loss_trigger_type(self) -> None:
        return None

    @property
    def take_profit_price(self) -> None:
        return None

    @property
    def take_profit_trigger_type(self) -> None:
        return None

    @property
    def time_in_force(self) -> None:
        return None

    @property
    def trailing_distance(self) -> None:
        return None


@dataclass(kw_only=True, slots=True)
class StopLossOrder(_RestingProtection):
    order_id: str = _get_order_id_field()
    kind: OrderType = OrderType.STOP_LOSS
    quantity: float | None
    side: OrderSide
    symbol: Symbol
    trigger_price: float
    trigger_type: TriggerType
    group_id: str
    reason: str | None
    client_order_id: str | None = None
    extra_fields: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"stop-loss order at {self.trigger_price} of {self.symbol}"


@dataclass(kw_only=True, slots=True)
class TrailingStopOrder(_RestingProtection):
    """A stop-loss that follows the best price its position has reached.

    `best_price` is the highest price the position has reached for a long and
    the lowest for a short, and the order starts from the price it is placed
    at. `trailing_share` is the share of `best_price` the stop stays behind it,
    which sets `trigger_price`, the level the stop rests at.
    """

    order_id: str = _get_order_id_field()
    kind: OrderType = OrderType.TRAILING_STOP
    quantity: float | None
    position_side: PositionSide
    symbol: Symbol
    best_price: float
    trailing_share: float
    reason: str | None
    client_order_id: str | None = None
    extra_fields: dict[str, Any] | None = None
    trigger_price: float = field(init=False)

    def __post_init__(self) -> None:
        self.trigger_price = trailing_level(
            self.position_side, self.best_price, self.trailing_share
        )

    @property
    def side(self) -> OrderSide:
        return self.position_side.closing_side

    @property
    def trigger_type(self) -> None:
        return None

    def trail(self, high: float, low: float) -> None:
        """Move the level to the share behind the best price a candle reached.

        The level follows the candle's high for a long and its low for a
        short, and a candle that does not beat the best price leaves it, so it
        never loosens.

        Args:
            high: The candle's high; NaN on a candle without a price.
            low: The candle's low; NaN on a candle without a price.
        """
        if self.position_side is PositionSide.LONG:
            if high > self.best_price:
                self.best_price = high
                self.trigger_price = trailing_level(
                    self.position_side, high, self.trailing_share
                )
        elif low < self.best_price:
            self.best_price = low
            self.trigger_price = trailing_level(
                self.position_side, low, self.trailing_share
            )

    def __str__(self) -> str:
        return (
            f"trailing-stop order at {self.trigger_price}, "
            f"{self.trailing_share:.2%} behind the price, of {self.symbol}"
        )


@dataclass(kw_only=True, slots=True)
class TakeProfitOrder(_RestingProtection):
    order_id: str = _get_order_id_field()
    kind: OrderType = OrderType.TAKE_PROFIT
    quantity: float | None
    side: OrderSide
    symbol: Symbol
    trigger_price: float
    trigger_type: TriggerType
    group_id: str
    reason: str | None
    client_order_id: str | None = None
    extra_fields: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"take-profit order at {self.trigger_price} of {self.symbol}"


@dataclass(kw_only=True, slots=True)
class TriggerOrder:
    """`rises_to_level` names the side of the level the order waits on, set
    from the close it is placed at, and `None` until a close of its symbol is
    known.
    """

    order_id: str = _get_order_id_field()
    kind: OrderType = OrderType.TRIGGER
    order: LimitOrder | MarketOrder
    trigger_price: float
    trigger_type: TriggerType
    rises_to_level: bool | None = None

    def __str__(self) -> str:
        return f"trigger order at {self.trigger_price} for order `{self.order}`"

    @property
    def symbol(self) -> Symbol:
        return self.order.symbol

    @property
    def side(self) -> OrderSide:
        return self.order.side

    @property
    def client_order_id(self) -> str | None:
        return self.order.client_order_id

    @property
    def limit_price(self) -> float | None:
        return self.order.limit_price

    @property
    def quantity(self) -> float:
        return self.order.quantity

    @property
    def reduce_only(self) -> bool:
        return self.order.reduce_only

    @property
    def remaining_quantity(self) -> float:
        return self.order.quantity

    @property
    def stop_loss_price(self) -> float | None:
        return self.order.stop_loss_price

    @property
    def stop_loss_trigger_type(self) -> TriggerType | None:
        return self.order.stop_loss_trigger_type

    @property
    def take_profit_price(self) -> float | None:
        return self.order.take_profit_price

    @property
    def take_profit_trigger_type(self) -> TriggerType | None:
        return self.order.take_profit_trigger_type

    @property
    def time_in_force(self) -> TimeInForce | None:
        return self.order.time_in_force

    @property
    def trailing_distance(self) -> None:
        return None


ExitOrder = (
    LimitOrder | MarketOrder | StopLossOrder | TakeProfitOrder | TrailingStopOrder
)
Order = ExitOrder | TriggerOrder


PROTECTION_ORDERS = (StopLossOrder, TakeProfitOrder, TrailingStopOrder)


def reported_reason(order: ExitOrder, named: str | None) -> str | None:
    """A stop-loss, a take-profit or a trailing stop acts on the position the
    strategy manages whatever placed it, and a trigger may belong to an order
    the strategy never placed. Both facts are `FillDescriber`'s contract.
    """
    if named is not None:
        return named
    return order.kind if isinstance(order, PROTECTION_ORDERS) else None
