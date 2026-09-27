import asyncio
import logging
from collections.abc import Awaitable, Iterable, Sequence

from robottraderslab._core import (
    OrderModifyRequest,
    OrderRequest,
    PlacedOrder,
    Symbol,
)
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ExchangeTransientError,
    StrategyCriticalError,
)

from .futures_exchange_protocol import FuturesExchangeProtocol

logger = logging.getLogger(__name__)


async def place_overlapped(
    venue: FuturesExchangeProtocol, requests: Sequence[OrderRequest]
) -> list[PlacedOrder | None]:
    """Place the orders one by one, overlapping the requests.

    Returns:
        The placed order for each request, in request order, with None where
        the venue rejected that request.
    """
    outcomes = await _overlapped(_place_one(venue, request) for request in requests)
    return _collect_placed([request.symbol for request in requests], outcomes)


async def place_in_order(
    venue: FuturesExchangeProtocol, requests: Sequence[OrderRequest]
) -> list[PlacedOrder | None]:
    """Place the orders one after another, for a venue that waits on nothing.

    Returns:
        The placed order for each request, in request order, with None where
        the venue rejected that request.
    """
    outcomes = await _in_order(_place_one(venue, request) for request in requests)
    return _collect_placed([request.symbol for request in requests], outcomes)


async def modify_overlapped(
    venue: FuturesExchangeProtocol, requests: Sequence[OrderModifyRequest]
) -> list[PlacedOrder | None]:
    """Cancel and re-place each resting order, overlapping the requests.

    Returns:
        The resulting order for each request, in request order, with None
        where the target was gone or the venue rejected the replacement.
    """
    resting = await _resting_order_ids(venue, requests)
    outcomes = await _overlapped(
        _modify_one(venue, request, resting) for request in requests
    )
    return _collect_placed([request.order.symbol for request in requests], outcomes)


async def modify_in_order(
    venue: FuturesExchangeProtocol, requests: Sequence[OrderModifyRequest]
) -> list[PlacedOrder | None]:
    """Cancel and re-place each resting order in turn, for a venue that waits on nothing.

    Returns:
        The resulting order for each request, in request order, with None
        where the target was gone or the venue rejected the replacement.
    """
    resting = await _resting_order_ids(venue, requests)
    outcomes = await _in_order(
        _modify_one(venue, request, resting) for request in requests
    )
    return _collect_placed([request.order.symbol for request in requests], outcomes)


async def cancel_overlapped(
    venue: FuturesExchangeProtocol, symbol: Symbol, order_ids: Iterable[str]
) -> None:
    """Cancel the orders together, every cancel settled before this returns.

    A venue refusing one cancel does not stop the others: the refusal is
    logged. A transient failure raises once every cancel has settled, so a
    retry reads the book with nothing still in flight.
    """
    outcomes = await _overlapped(
        venue.cancel_order_by_id(symbol, order_id) for order_id in order_ids
    )
    for outcome in outcomes:
        _accepted(symbol, outcome, "Cancel")


async def _overlapped[T](calls: Iterable[Awaitable[T]]) -> list[T | BaseException]:
    return await asyncio.gather(*(_capture(call) for call in calls))


async def _in_order[T](calls: Iterable[Awaitable[T]]) -> list[T | BaseException]:
    return [await _capture(call) for call in calls]


async def _capture[T](call: Awaitable[T]) -> T | BaseException:
    try:
        return await call
    except (Exception, ExchangeCriticalError, StrategyCriticalError) as error:
        return error


async def _resting_order_ids(
    venue: FuturesExchangeProtocol, requests: Sequence[OrderModifyRequest]
) -> set[str]:
    orders = await venue.get_open_orders([request.order.symbol for request in requests])
    return {order.order_id for order in orders}


def _collect_placed(
    symbols: Sequence[Symbol],
    outcomes: Sequence[PlacedOrder | None | BaseException],
) -> list[PlacedOrder | None]:
    return [
        _accepted(symbol, outcome, "Order")
        for symbol, outcome in zip(symbols, outcomes)
    ]


def _accepted[T](symbol: Symbol, outcome: T | BaseException, request: str) -> T | None:
    if isinstance(
        outcome,
        (ExchangeCriticalError, StrategyCriticalError, ExchangeTransientError),
    ):
        raise outcome
    if isinstance(outcome, ExchangeRecoverableError):
        logger.warning("%s rejected in batch (%s): %s", request, symbol, outcome)
        return None
    if isinstance(outcome, BaseException):
        logger.error("Internal error in batch (%s): %s", symbol, outcome)
        return None
    return outcome


async def _modify_one(
    venue: FuturesExchangeProtocol,
    request: OrderModifyRequest,
    open_order_ids: set[str],
) -> PlacedOrder | None:
    if request.order_id not in open_order_ids:
        logger.warning(
            "Order %s (%s) is no longer resting; skipping its modification",
            request.order_id,
            request.order.symbol,
        )
        return None
    await venue.cancel_order_by_id(request.order.symbol, request.order_id)
    return await _place_one(venue, request.order)


async def _place_one(
    venue: FuturesExchangeProtocol, request: OrderRequest
) -> PlacedOrder:
    if request.limit_price is None:
        return await venue.place_market_order(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            reduce_only=request.reduce_only,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            trigger_price=request.trigger_price,
            client_order_id=request.client_order_id,
        )
    return await venue.place_limit_order(
        symbol=request.symbol,
        side=request.side,
        quantity=request.quantity,
        price=request.limit_price,
        reduce_only=request.reduce_only,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
        trigger_price=request.trigger_price,
        client_order_id=request.client_order_id,
        time_in_force=request.time_in_force,
    )
