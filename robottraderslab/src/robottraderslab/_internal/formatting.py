import math

FRACTION_TO_PERCENT = 100.0
_INFINITY = "∞"


def format_percent(fraction: float) -> str:
    return f"{fraction * FRACTION_TO_PERCENT:.2f}%"


def format_ratio(value: float) -> str:
    """A ratio with no losses to divide by is infinite, and reads as such."""
    if math.isinf(value):
        return _INFINITY
    return f"{value:.2f}"


def format_money(amount: float) -> str:
    """The sign leads the currency symbol, so a loss reads as one figure."""
    if amount < 0.0:
        return f"-${abs(amount):.2f}"
    return f"${amount:.2f}"
