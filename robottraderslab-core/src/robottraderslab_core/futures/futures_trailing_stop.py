from collections.abc import Awaitable, Callable
from decimal import ROUND_HALF_UP, Decimal
from functools import partial

from robottraderslab_core._internal import OrderType, PlacedOrder, Symbol
from robottraderslab_core.exceptions import BotCriticalError

from .futures_capabilities import FuturesCapabilities
from .futures_exchange_protocol import FuturesExchangeProtocol

_PERCENT = Decimal(100)
_PERCENT_STEP = Decimal("0.01")


async def replace_trailing_stops(
    venue: FuturesExchangeProtocol,
    symbol: Symbol,
    distance: float,
    activation_price: float | None,
    quantity: float | None,
) -> PlacedOrder:
    """Place the new trailing stop, then cancel every other one resting on the symbol.

    The distance is in the quote currency and the quantity in the units of the
    position's quantity. The position holds a trailing stop at every moment of
    the call, and a placement that raises leaves the earlier ones resting.

    Args:
        distance: How far, in the quote currency, the new order stays behind
            the best price reached.
        activation_price: Price the market reaches before the trail starts.
        quantity: Size the new order closes, in the units of the position's
            quantity.

    Returns:
        The new trailing stop, under the id it reports among its open orders.
    """
    return await replace_trailing_stops_with(
        venue,
        symbol,
        partial(
            venue.place_trailing_stop_order,
            symbol,
            distance,
            activation_price,
            quantity,
        ),
    )


async def replace_trailing_stops_with(
    venue: FuturesExchangeProtocol,
    symbol: Symbol,
    place: Callable[[], Awaitable[PlacedOrder]],
) -> PlacedOrder:
    """Place a trailing stop, then cancel every other one resting on the symbol.

    The position holds a trailing stop at every moment of the call, and a
    placement that raises leaves the earlier ones resting.

    Args:
        place: Places the new trailing stop.

    Returns:
        The new trailing stop, under the id it reports among its open orders.
    """
    placed = await place()
    for order in await venue.get_open_orders([symbol]):
        if (
            order.kind is OrderType.TRAILING_STOP
            and order.symbol == symbol
            and order.order_id != placed.order_id
        ):
            await venue.cancel_order_by_id(symbol, order.order_id)
    return placed


def trailing_percent(
    capabilities: FuturesCapabilities, symbol: Symbol, share: float
) -> str:
    """Return a share of the price as the percentage, at two decimals, a venue takes.

    Args:
        capabilities: What the venue declares, the band of shares it holds a
            trailing stop at included.
        share: The share of the price, 0.02 for 2%.

    Raises:
        BotCriticalError: If the venue holds no trailing stop at the share.
    """
    if not capabilities.holds_trailing_share(share):
        raise BotCriticalError(
            f"{symbol}: a trailing share of {share} is outside the "
            f"{_percent_text(capabilities.min_trailing_share)}% to "
            f"{_percent_text(capabilities.max_trailing_share)}% of the price the "
            "venue takes"
        )
    return str(_percent_of(share).quantize(_PERCENT_STEP, rounding=ROUND_HALF_UP))


def _percent_of(share: float) -> Decimal:
    return Decimal(str(share)) * _PERCENT


def _percent_text(share: float) -> str:
    return format(_percent_of(share).normalize(), "f")
