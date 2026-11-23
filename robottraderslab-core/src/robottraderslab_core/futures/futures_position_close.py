from robottraderslab_core._internal import (
    PlacedOrder,
    PositionSnapshot,
    Symbol,
    validate_positive,
)
from robottraderslab_core.exceptions import (
    ExchangeRecoverableError,
    NoOpenPositionError,
)

from .futures_exchange_protocol import FuturesExchangeProtocol


async def close_held_position(
    venue: FuturesExchangeProtocol, symbol: Symbol, quantity: float | None = None
) -> PlacedOrder:
    """Close the position the account holds on the symbol, or a quantity of it, at market.

    The position is read once and one reduce-only market order on its opposite
    side is placed, whichever connector or simulator the venue is.

    Args:
        venue: The exchange whose position read and market order run.
        symbol: The symbol whose position closes.
        quantity: Size to close, in the units of the position's quantity; none
            closes the whole position.

    Returns:
        The order placed.

    Raises:
        NoOpenPositionError: If the account holds no position on the symbol; no
            order is placed.
        ExchangeRecoverableError: If the quantity is above the position's size;
            no order is placed.
        ValueError: If the quantity is not above zero; no order is placed.
    """
    position = await held_position(venue, symbol)
    if position is None:
        raise NoOpenPositionError(f"No open position on {symbol} to close")
    size = position.quantity if quantity is None else _validated(position, quantity)
    return await venue.place_market_order(
        symbol, position.side.closing_side, size, reduce_only=True
    )


async def held_position(
    venue: FuturesExchangeProtocol, symbol: Symbol
) -> PositionSnapshot | None:
    """Read the account's positions on the symbol and return the one it holds, or none."""
    return (await venue.get_open_positions([symbol])).get(symbol)


def _validated(position: PositionSnapshot, quantity: float) -> float:
    validate_positive("quantity", quantity)
    if quantity > position.quantity:
        raise ExchangeRecoverableError(
            f"{position.symbol}: `quantity` must be at most the position's "
            f"{position.quantity}; received {quantity}"
        )
    return quantity
