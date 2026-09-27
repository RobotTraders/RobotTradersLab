def calculate_pnl_long(
    entry_price: float, close_price: float, closed_quantity: float
) -> float:
    """
    Calculates realized PnL for long positions.

    Args:
        entry_price: Average entry price
        close_price: Current market price
        closed_quantity: Quantity being closed (in base currency)

    Returns:
        Realized PnL in quote currency
    """
    return (close_price - entry_price) * closed_quantity


def calculate_pnl_short(
    entry_price: float, close_price: float, closed_quantity: float
) -> float:
    """
    Calculates realized PnL for short positions.

    Args:
        entry_price: Average entry price
        close_price: Current market price
        closed_quantity: Quantity being closed (in base currency)

    Returns:
        Realized PnL in quote currency
    """
    return (entry_price - close_price) * closed_quantity


def calculate_trade_pnl_pct(
    pnl: float, entry_price: float, closed_quantity: float
) -> float:
    """
    Calculates percentage ROI for a trade.

    Args:
        pnl: Realized PnL in quote currency
        entry_price: Average entry price
        closed_quantity: Quantity being closed (in base currency)

    Returns:
        PnL as a percentage of the initial position value
    """
    return pnl / (entry_price * closed_quantity)
