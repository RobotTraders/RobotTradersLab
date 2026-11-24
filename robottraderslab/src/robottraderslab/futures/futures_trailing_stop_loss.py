from robottraderslab._internal import is_trailing_share


def validate_trailing_ratio(distance_ratio: float) -> None:
    """A trailing stop sits some distance from the price and never past it.

    Raises:
        ValueError: If the ratio is outside (0, 1) or is NaN.
    """
    if not is_trailing_share(distance_ratio):
        raise ValueError(
            f"`distance_ratio` must be within (0, 1); received {distance_ratio}"
        )
