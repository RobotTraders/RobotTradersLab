from robottraderslab_core.exchanges import PositionSide


def is_trailing_share(share: float) -> bool:
    """Report whether a share of the price is one a trailing stop trails at.

    A trailing stop sits some distance from the price and never past it, so
    the share lies within (0, 1); NaN is none.
    """
    return 0.0 < share < 1.0


def trailing_level(side: PositionSide, best_price: float, share: float) -> float:
    """Return the level a trailing stop rests at behind the best price.

    Args:
        side: The side of the position the stop guards.
        best_price: The highest price the position has reached for a long, the
            lowest for a short.
        share: The share of `best_price` the stop stays behind it, within
            (0, 1).

    Returns:
        The price the share below `best_price` for a long, above it for a
        short.
    """
    if side is PositionSide.LONG:
        return best_price * (1 - share)
    return best_price * (1 + share)
