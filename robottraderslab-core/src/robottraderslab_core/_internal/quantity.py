STEPS_PER_UNIT = 100_000_000

type StepCount = int


def to_step_count(quantity: float) -> StepCount:
    """A quantity a venue or the simulator reported is a whole number of steps
    and comes back as exactly that count.
    """
    return round(quantity * STEPS_PER_UNIT)


def to_quantity(steps: StepCount) -> float:
    """What leaves the engine is a quantity a venue can fill."""
    return steps / STEPS_PER_UNIT
