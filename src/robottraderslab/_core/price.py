import math


def validate_price(name: str, value: float) -> None:
    """A price is a number above zero.

    Raises:
        ValueError: If the value is not positive or is NaN.
    """
    if value <= 0 or math.isnan(value):
        raise ValueError(f"`{name}` must be greater than 0; received {value}")
