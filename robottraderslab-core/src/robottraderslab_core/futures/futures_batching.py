import asyncio
import logging
from collections.abc import Awaitable, Iterable, Mapping, Sequence

from robottraderslab_core._internal import (
    OrderModifyRequest,
    OrderProtocol,
    OrderRequest,
    PlacedOrder,
    Symbol,
    describe_order,
    report_refusal,
    tag_of,
    with_resting_reduce_only,
)
from robottraderslab_core.exceptions import (
    BotCriticalError,
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ExchangeTransientError,
)

from .futures_exchange_protocol import FuturesExchangeProtocol

logger = logging.getLogger(__name__)


def report_order_refusal(
    request: OrderRequest, error: ExchangeRecoverableError
) -> None:
    """Report a refusal of one order of a batch at the level it owes.

    A refused exit is an ERROR, an answer that no position is held is an INFO,
    an entry refused for lack of margin joins the cycle's single line when its
    call reports margin refusals together and is its own WARNING otherwise,
    and any other refusal is a WARNING.

    Args:
        request: The order the venue refused; its client order id's tag names
            the profile and `is_exit` says whether it closes exposure.
        error: The venue's answer.
    """
    report_refusal(
        request.symbol,
        tag_of(request.client_order_id),
        error,
        is_exit=request.is_exit,
        what=_described(request),
    )


async def place_overlapped(
    venue: FuturesExchangeProtocol, requests: Sequence[OrderRequest]
) -> list[PlacedOrder | None]:
    """Place the orders one by one, overlapping the requests.

    Returns:
        The placed order for each request, in request order, with None where
        the venue rejected that request.
    """
    outcomes = await _overlapped(_place_one(venue, request) for request in requests)
    return _collect_placed(requests, outcomes)


async def place_in_order(
    venue: FuturesExchangeProtocol, requests: Sequence[OrderRequest]
) -> list[PlacedOrder | None]:
    """Place the orders one after another, for a venue that waits on nothing.

    Returns:
        The placed order for each request, in request order, with None where
        the venue rejected that request.
    """
    outcomes = await _in_order(_place_one(venue, request) for request in requests)
    return _collect_placed(requests, outcomes)


async def modify_overlapped(
    venue: FuturesExchangeProtocol, requests: Sequence[OrderModifyRequest]
) -> list[PlacedOrder | None]:
    """Cancel and re-place each resting order, overlapping the requests.

    Returns:
        The resulting order for each request, in request order, with None
        where the target was gone or the venue rejected the replacement.
    """
    resting = await _resting_orders(venue, requests)
    outcomes = await _overlapped(
        _modify_one(venue, request, resting) for request in requests
    )
    return _collect_modified(requests, outcomes)


async def modify_in_order(
    venue: FuturesExchangeProtocol, requests: Sequence[OrderModifyRequest]
) -> list[PlacedOrder | None]:
    """Cancel and re-place each resting order in turn, for a venue that waits on nothing.

    Returns:
        The resulting order for each request, in request order, with None
        where the target was gone or the venue rejected the replacement.
    """
    resting = await _resting_orders(venue, requests)
    outcomes = await _in_order(
        _modify_one(venue, request, resting) for request in requests
    )
    return _collect_modified(requests, outcomes)


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


def replacement_orders(requests: Iterable[OrderModifyRequest]) -> list[OrderRequest]:
    """Return the replacement each modification request asks the venue to place."""
    return [request.order for request in requests]


def _described(request: OrderRequest) -> str:
    described = describe_order(request)
    if request.is_exit:
        return f"{described}, part of an exit"
    return described


async def _overlapped[T](calls: Iterable[Awaitable[T]]) -> list[T | BaseException]:
    return await asyncio.gather(*(_capture(call) for call in calls))


async def _in_order[T](calls: Iterable[Awaitable[T]]) -> list[T | BaseException]:
    return [await _capture(call) for call in calls]


async def _capture[T](call: Awaitable[T]) -> T | BaseException:
    try:
        return await call
    except (Exception, ExchangeCriticalError, BotCriticalError) as error:
        return error


async def _resting_orders(
    venue: FuturesExchangeProtocol, requests: Sequence[OrderModifyRequest]
) -> dict[str, OrderProtocol]:
    orders = await venue.get_open_orders(
        [request.order.symbol for request in requests], with_triggers=True
    )
    return {order.order_id: order for order in orders}


def _collect_placed(
    requests: Sequence[OrderRequest],
    outcomes: Sequence[PlacedOrder | None | BaseException],
) -> list[PlacedOrder | None]:
    return [_placed(request, outcome) for request, outcome in zip(requests, outcomes)]


def _collect_modified(
    requests: Sequence[OrderModifyRequest],
    outcomes: Sequence[PlacedOrder | None | BaseException],
) -> list[PlacedOrder | None]:
    return _collect_placed(replacement_orders(requests), outcomes)


def _placed(
    request: OrderRequest, outcome: PlacedOrder | None | BaseException
) -> PlacedOrder | None:
    if isinstance(outcome, ExchangeRecoverableError) and not isinstance(
        outcome, ExchangeTransientError
    ):
        report_order_refusal(request, outcome)
        return None
    return _accepted(request.symbol, outcome, "Order")


def _accepted[T](symbol: Symbol, outcome: T | BaseException, request: str) -> T | None:
    if isinstance(
        outcome,
        (ExchangeCriticalError, BotCriticalError, ExchangeTransientError),
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
    open_orders: Mapping[str, OrderProtocol],
) -> PlacedOrder | None:
    resting = open_orders.get(request.order_id)
    if resting is None:
        logger.warning(
            "Order %s (%s) is no longer resting; skipping its modification",
            request.order_id,
            request.order.symbol,
        )
        return None
    replacement = with_resting_reduce_only(request, resting.reduce_only)
    await venue.cancel_order_by_id(request.order.symbol, request.order_id)
    return await _place_one(venue, replacement)


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
            trigger_type=request.trigger_type,
            client_order_id=request.client_order_id,
            params=request.params,
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
        trigger_type=request.trigger_type,
        client_order_id=request.client_order_id,
        time_in_force=request.time_in_force,
        params=request.params,
    )
