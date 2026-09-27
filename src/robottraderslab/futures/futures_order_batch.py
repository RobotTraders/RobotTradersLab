from dataclasses import dataclass

from robottraderslab._core import (
    ActionBuilder,
    ActionResult,
    BaseExchangeAction,
    OrderModifyRequest,
    OrderOutcome,
    OrderPlacement,
    OrderRequest,
    TimeInForce,
)

from .futures_exchange_protocol import FuturesExchangeProtocol
from .futures_limit_order import FuturesLimitOrderAction
from .futures_market_order import FuturesMarketOrderAction

type BatchableOrderAction = FuturesLimitOrderAction | FuturesMarketOrderAction
type BatchableOrder = BatchableOrderAction | ActionBuilder[BatchableOrderAction]


@dataclass(kw_only=True, slots=True)
class FuturesOrderBatchAction(BaseExchangeAction):
    """Several orders placed together, as few requests as the venue allows.

    Attributes:
        symbol: Symbol of the first order, standing in for the batch in logs.
        orders: Order actions in submission order; the venue's answers are
            matched to them by position.
    """

    exchange: FuturesExchangeProtocol
    orders: list[BatchableOrderAction]

    async def execute(self) -> ActionResult:
        placed = await self.exchange.place_orders(
            [_to_request(order) for order in self.orders]
        )
        return ActionResult(
            orders=tuple(
                OrderOutcome(
                    placed_order=placed_order,
                    on_filled=order.on_filled,
                    placement=_placement_of(order, placed_order.order_id),
                )
                for order, placed_order in zip(self.orders, placed)
                if placed_order is not None
            )
        )


@dataclass(kw_only=True, frozen=True, slots=True)
class OrderModification:
    """A resting order paired with the replacement it should become.

    Attributes:
        order_id: Exchange-assigned id of the resting order.
        order: The replacement, a built order or an order builder the
            account finishes with its `build()`.
    """

    order_id: str
    order: BatchableOrder


@dataclass(kw_only=True, frozen=True, slots=True)
class OrderReplacement:
    """A resting order paired with the finished order it becomes.

    Attributes:
        order_id: Exchange-assigned id of the resting order.
        order: The finished order it becomes.
    """

    order_id: str
    order: BatchableOrderAction


@dataclass(kw_only=True, slots=True)
class FuturesOrderModifyAction(BaseExchangeAction):
    """Several resting orders reshaped together, as few requests as the venue allows.

    Attributes:
        symbol: Symbol of the first modification, standing in for the batch in logs.
    """

    exchange: FuturesExchangeProtocol
    modifications: list[OrderReplacement]

    async def execute(self) -> ActionResult:
        placed = await self.exchange.modify_orders(
            [
                OrderModifyRequest(
                    order_id=modification.order_id,
                    order=_to_request(modification.order),
                )
                for modification in self.modifications
            ]
        )
        return ActionResult(
            orders=tuple(
                OrderOutcome(
                    placed_order=placed_order,
                    on_filled=modification.order.on_filled,
                    placement=_placement_of(modification.order, placed_order.order_id),
                )
                for modification, placed_order in zip(self.modifications, placed)
                if placed_order is not None
            )
        )


def _placement_of(order: BatchableOrderAction, order_id: str) -> OrderPlacement:
    return OrderPlacement(
        order_id=order_id,
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        kind=order.kind,
        price=_limit_price_of(order),
        trigger_price=order.trigger_price,
        client_order_id=order.client_order_id,
        reason=order.reason,
    )


def _limit_price_of(order: BatchableOrderAction) -> float | None:
    return order.price if isinstance(order, FuturesLimitOrderAction) else None


def _time_in_force_of(order: BatchableOrderAction) -> TimeInForce:
    if isinstance(order, FuturesLimitOrderAction):
        return order.time_in_force
    return TimeInForce.GTC


def _to_request(order: BatchableOrderAction) -> OrderRequest:
    limit_price = _limit_price_of(order)
    return OrderRequest(
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        limit_price=limit_price,
        time_in_force=_time_in_force_of(order),
        reduce_only=order.reduce_only,
        stop_loss=order.stop_loss,
        take_profit=order.take_profit,
        trigger_price=order.trigger_price,
        client_order_id=order.client_order_id,
    )
