import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial

from robottraderslab_core._internal import (
    ActionBuilder,
    ActionResult,
    BaseExchangeAction,
    OrderModifyRequest,
    OrderOutcome,
    OrderPlacement,
    OrderRequest,
    PlacedOrder,
    Symbol,
    TimeInForce,
    report_refusal,
)
from robottraderslab_core.exceptions import (
    ExchangeRecoverableError,
    ExchangeTransientError,
    InsufficientMarginError,
)

from .futures_batching import (
    cancel_overlapped,
    replacement_orders,
    report_order_refusal,
)
from .futures_exchange_protocol import FuturesExchangeProtocol
from .futures_limit_order import FuturesLimitOrderAction
from .futures_market_order import FuturesMarketOrderAction

type BatchableOrderAction = FuturesLimitOrderAction | FuturesMarketOrderAction
type BatchableOrder = BatchableOrderAction | ActionBuilder[BatchableOrderAction]

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class FuturesOrderBatchAction(BaseExchangeAction):
    """Several orders placed together, as few requests as the venue allows.

    An order the venue's trading rules denied is left out and the others are
    placed: a denied entry is a WARNING, a denied exit an ERROR naming it as a
    refused exit, each reported once however often the batch is retried.

    Attributes:
        symbol: Symbol of the first order, standing in for the batch in logs.
        orders: Order actions in submission order; the venue's answers are
            matched to them by position.
    """

    exchange: FuturesExchangeProtocol
    orders: list[BatchableOrderAction]
    _settled: set[str] = field(default_factory=set, init=False, repr=False)

    async def execute(self) -> ActionResult:
        orders = self.orders
        requests = [_to_request(order) for order in orders if order.denial is None]
        if len(requests) != len(orders):
            await _once(self._settled, "report", partial(_log_denied, self.orders))
            if not requests:
                return ActionResult()
            orders = [order for order in orders if order.denial is None]
        placed = await _placed_or_reported(
            self.exchange.place_orders(requests), requests
        )
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
    """An order this bot did not place is replaced with no client order id, so
    the venue draws one.
    """

    order_id: str
    order: BatchableOrderAction
    with_client_order_id: bool = True


@dataclass(kw_only=True, slots=True)
class FuturesOrderModifyAction(BaseExchangeAction):
    """Several resting orders reshaped together, as few requests as the venue allows.

    A resting order whose replacement the venue's trading rules denied is
    cancelled, since the order the strategy asked for cannot rest; a denied
    exit replacement is also an ERROR naming it as a refused exit. However
    often the batch is retried, each denial is reported once, and the
    cancels of the resting orders are all sent again on each retry until
    one attempt settles every one of them.

    Attributes:
        symbol: Symbol of the first modification, standing in for the batch in logs.
    """

    exchange: FuturesExchangeProtocol
    modifications: list[OrderReplacement]
    _settled: set[str] = field(default_factory=set, init=False, repr=False)

    async def execute(self) -> ActionResult:
        modifications = self.modifications
        requests = _modify_requests(modifications)
        if len(requests) != len(modifications):
            await self._settle_denied_once()
            modifications = [
                modification
                for modification in modifications
                if modification.order.denial is None
            ]
            if not modifications:
                return ActionResult()
        placed = await _placed_or_reported(
            self.exchange.modify_orders(requests),
            replacement_orders(requests),
        )
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

    async def _settle_denied_once(self) -> None:
        report = partial(_report_denied, self.modifications)
        await _once(self._settled, "report", report)
        cancel = partial(_cancel_denied, self.exchange, self.modifications)
        await _once(self._settled, "cancel", cancel)


def _to_request(
    order: BatchableOrderAction, *, with_client_order_id: bool = True
) -> OrderRequest:
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
        client_order_id=order.client_order_id if with_client_order_id else None,
        params=order.params,
        is_exit=order.is_exit,
    )


async def _once(
    settled: set[str], step: str, work: Callable[[], Awaitable[None] | None]
) -> None:
    if step in settled:
        return
    outcome = work()
    if outcome is not None:
        await outcome
    settled.add(step)


def _log_denied(orders: Sequence[BatchableOrderAction]) -> None:
    _report_denied_exits(orders)
    for order in orders:
        if order.denial is not None and not order.is_exit:
            logger.warning(
                "Order left out of its batch (%s): %s", order.symbol, order.denial
            )


def _report_denied_exits(orders: Iterable[BatchableOrderAction]) -> None:
    for order in orders:
        if order.denial is not None and order.is_exit:
            report_refusal(
                order.symbol,
                order.tag,
                ExchangeRecoverableError(order.denial),
                is_exit=True,
                what=type(order).__name__,
            )


async def _placed_or_reported(
    send: Awaitable[list[PlacedOrder | None]], requests: Sequence[OrderRequest]
) -> list[PlacedOrder | None]:
    try:
        return await send
    except ExchangeTransientError:
        raise
    except ExchangeRecoverableError as error:
        if not _is_reported_per_order(requests, error):
            raise
        for request in requests:
            report_order_refusal(request, error)
        return [None] * len(requests)


def _is_reported_per_order(
    requests: Sequence[OrderRequest], error: ExchangeRecoverableError
) -> bool:
    return isinstance(error, InsufficientMarginError) or any(
        request.is_exit for request in requests
    )


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


def _modify_requests(
    modifications: Iterable[OrderReplacement],
) -> list[OrderModifyRequest]:
    return [
        OrderModifyRequest(
            order_id=modification.order_id,
            order=_to_request(
                modification.order,
                with_client_order_id=modification.with_client_order_id,
            ),
        )
        for modification in modifications
        if modification.order.denial is None
    ]


def _report_denied(modifications: Sequence[OrderReplacement]) -> None:
    _report_denied_exits([modification.order for modification in modifications])
    for modification in modifications:
        order = modification.order
        if order.denial is not None:
            logger.warning(
                "%s: the order resting as %s is cancelled, since its replacement "
                "is denied: %s",
                order.symbol,
                modification.order_id,
                order.denial,
            )


async def _cancel_denied(
    exchange: FuturesExchangeProtocol, modifications: Iterable[OrderReplacement]
) -> None:
    order_ids: dict[Symbol, list[str]] = {}
    for modification in modifications:
        if modification.order.denial is not None:
            order_ids.setdefault(modification.order.symbol, []).append(
                modification.order_id
            )
    for symbol, ids in order_ids.items():
        await cancel_overlapped(exchange, symbol, ids)
