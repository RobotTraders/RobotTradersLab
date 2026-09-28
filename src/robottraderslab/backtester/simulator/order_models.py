from dataclasses import dataclass, field
from itertools import count
from typing import Any

from robottraderslab._core import Symbol
from robottraderslab.exchanges import (
    OrderSide,
    OrderType,
    StopLoss,
    TakeProfit,
    TimeInForce,
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
    client_order_id: str | None = None
    extra_fields: dict[str, Any] | None = None
    triggered: bool = False


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

    def __str__(self) -> str:
        return f"market order {self.side} {self.quantity:,.6f} of {self.symbol}"


@dataclass(kw_only=True, slots=True)
class StopLossOrder:
    order_id: str = _get_order_id_field()
    kind: OrderType = OrderType.STOP_LOSS
    quantity: float | None = None
    side: OrderSide
    symbol: Symbol
    trigger_price: float
    group_id: str
    reason: str | None
    client_order_id: str | None = None
    extra_fields: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"stop-loss order at {self.trigger_price} of {self.symbol}"


@dataclass(kw_only=True, slots=True)
class TakeProfitOrder:
    order_id: str = _get_order_id_field()
    kind: OrderType = OrderType.TAKE_PROFIT
    quantity: float | None = None
    side: OrderSide
    symbol: Symbol
    trigger_price: float
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


Order = LimitOrder | MarketOrder | StopLossOrder | TakeProfitOrder | TriggerOrder


_PROTECTION_ORDERS = (StopLossOrder, TakeProfitOrder)


def reported_reason(
    order: LimitOrder | MarketOrder | StopLossOrder | TakeProfitOrder,
    named: str | None,
) -> str | None:
    """A stop-loss or a take-profit acts on the position the strategy manages
    whatever placed it, and a trigger may belong to an order the strategy never
    placed. Both facts are `FillDescriber`'s contract.
    """
    if named is not None:
        return named
    return order.kind if isinstance(order, _PROTECTION_ORDERS) else None
