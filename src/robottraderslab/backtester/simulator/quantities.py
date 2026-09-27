from robottraderslab._core import StepCount, to_quantity, to_step_count


def to_floored_step_count(quantity: float) -> StepCount:
    """A venue cuts an order down to its lot and never up; a quantity read
    back from a position is already whole and loses nothing.
    """
    steps = to_step_count(quantity)
    if to_quantity(steps) > quantity:
        return steps - 1
    return steps
