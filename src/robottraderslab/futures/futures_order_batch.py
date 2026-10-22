import logging
from collections.abc import Iterable
from dataclasses import dataclass

from robottraderslab._core import (
    ActionBuilder,
    ActionResult,
    BaseExchangeAction,
    OrderModifyRequest,
    OrderOutcome,
    OrderPlacement,
    OrderRequest,
    Symbol,
    TimeInForce,
)

from .futures_batching import cancel_overlapped
from .futures_exchange_protocol import FuturesExchangeProtocol
from .futures_limit_order import FuturesLimitOrderAction
from .futures_market_order import FuturesMarketOrderAction

type BatchableOrderAction = FuturesLimitOrderAction | FuturesMarketOrderAction
type BatchableOrder = BatchableOrderAction | ActionBuilder[BatchableOrderAction]

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class FuturesOrderBatchAction(BaseExchangeAction):
    """Several orders placed together, as few requests as the venue allows.

    An order the venue's trading rules denied is left out with a warning, and
    the others are placed.

    Attributes:
        symbol: Symbol of the first order, standing in for the batch in logs.
        orders: Order actions in submission order; the venue's answers are
            matched to them by position.
    """

    exchange: FuturesExchangeProtocol
    orders: list[BatchableOrderAction]

    async def execute(self) -> ActionResult:
        orders = self.orders
        requests = [_to_request(order) for order in orders if order.denial is None]
        if len(requests) != len(orders):
            _log_denied(orders)
            if not requests:
                return ActionResult()
            orders = [order for order in orders if order.denial is None]
        placed = await self.exchange.place_orders(requests)
        return ActionResult(
            orders=tuple(
                OrderOutcome(
                    placed_order=placed_order,
                    on_filled=order.on_filled,
                    placement=_placement_of(order, request, placed_order.order_id),
                )
                for order, request, placed_order in zip(orders, requests, placed)
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

    A resting order whose replacement the venue's trading rules denied is
    cancelled, since the order the strategy asked for cannot rest.

    Attributes:
        symbol: Symbol of the first modification, standing in for the batch in logs.
    """

    exchange: FuturesExchangeProtocol
    modifications: list[OrderReplacement]

    async def execute(self) -> ActionResult:
        modifications = self.modifications
        requests = [
            OrderModifyRequest(
                order_id=modification.order_id,
                order=_to_request(modification.order),
            )
            for modification in modifications
            if modification.order.denial is None
        ]
        if len(requests) != len(modifications):
            await _cancel_denied(self.exchange, modifications)
            if not requests:
                return ActionResult()
            modifications = [
                modification
                for modification in modifications
                if modification.order.denial is None
            ]
        placed = await self.exchange.modify_orders(requests)
        return ActionResult(
            orders=tuple(
                OrderOutcome(
                    placed_order=placed_order,
                    on_filled=modification.order.on_filled,
                    placement=_placement_of(
                        modification.order, request.order, placed_order.order_id
                    ),
                )
                for modification, request, placed_order in zip(
                    modifications, requests, placed
                )
                if placed_order is not None
            )
        )


def _log_denied(orders: Iterable[BatchableOrderAction]) -> None:
    for order in orders:
        if order.denial is not None:
            logger.warning(
                "Order left out of its batch (%s): %s", order.symbol, order.denial
            )


async def _cancel_denied(
    exchange: FuturesExchangeProtocol, modifications: Iterable[OrderReplacement]
) -> None:
    order_ids: dict[Symbol, list[str]] = {}
    for modification in modifications:
        order = modification.order
        if order.denial is None:
            continue
        logger.warning(
            "%s: the order resting as %s is cancelled, since its replacement is "
            "denied: %s",
            order.symbol,
            modification.order_id,
            order.denial,
        )
        order_ids.setdefault(order.symbol, []).append(modification.order_id)
    for symbol, ids in order_ids.items():
        await cancel_overlapped(exchange, symbol, ids)


def _placement_of(
    order: BatchableOrderAction, request: OrderRequest, order_id: str
) -> OrderPlacement:
    return OrderPlacement(
        order_id=order_id,
        symbol=request.symbol,
        side=request.side,
        quantity=request.quantity,
        kind=order.kind,
        price=request.limit_price,
        trigger_price=request.trigger_price,
        client_order_id=request.client_order_id,
        reason=order.reason,
    )


def _to_request(order: BatchableOrderAction) -> OrderRequest:
    if isinstance(order, FuturesLimitOrderAction):
        limit_price, time_in_force = order.price, order.time_in_force
    else:
        limit_price, time_in_force = None, TimeInForce.GTC
    return OrderRequest(
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        limit_price=limit_price,
        time_in_force=time_in_force,
        reduce_only=order.reduce_only,
        stop_loss=order.stop_loss,
        take_profit=order.take_profit,
        trigger_price=order.trigger_price,
        trigger_type=order.trigger_type,
        client_order_id=order.client_order_id,
        params=order.params,
    )
