from datetime import datetime

import pytest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import SimulatedPosition
from robottraderslab.exchanges import PositionSide

FIRST_FILL_STEPS = 2_998_500
SECOND_FILL_STEPS = 4_997_500
THIRD_FILL_STEPS = 1_999_000
HALF_OF_SECOND_FILL_STEPS = SECOND_FILL_STEPS // 2
A_HUNDRED_MILLION_UNITS_IN_STEPS = 10**16
ONE_STEP = 1
PRICE = 2500.0


@pytest.fixture
def position() -> SimulatedPosition:
    return SimulatedPosition(
        symbol=Symbol.create("ETH/USDT:USDT"),
        side=PositionSide.LONG,
        leverage=1.0,
        taker_fee_rate=0.0,
        entry_time=datetime(2026, 1, 1),
    )


class TestStepAccumulation:
    @pytest.mark.parametrize(
        ("fills", "removals"),
        [
            (
                (FIRST_FILL_STEPS, SECOND_FILL_STEPS),
                (FIRST_FILL_STEPS, SECOND_FILL_STEPS),
            ),
            (
                (FIRST_FILL_STEPS, SECOND_FILL_STEPS),
                (SECOND_FILL_STEPS, FIRST_FILL_STEPS),
            ),
            (
                (FIRST_FILL_STEPS, SECOND_FILL_STEPS, THIRD_FILL_STEPS),
                (SECOND_FILL_STEPS, THIRD_FILL_STEPS, FIRST_FILL_STEPS),
            ),
        ],
    )
    def test_fills_removed_in_any_order_leave_nothing(self, position, fills, removals):
        for steps in fills:
            position.add_to_position(steps, PRICE)

        for steps in removals:
            position.reduce_position(steps)

        assert position.quantity_steps == 0

    def test_a_partial_reduce_leaves_the_other_fill_whole(self, position):
        position.add_to_position(FIRST_FILL_STEPS, PRICE)
        position.add_to_position(SECOND_FILL_STEPS, PRICE)

        position.reduce_position(HALF_OF_SECOND_FILL_STEPS)
        position.reduce_position(HALF_OF_SECOND_FILL_STEPS)

        assert position.quantity_steps == FIRST_FILL_STEPS

    def test_one_step_beside_a_hundred_million_units_can_be_removed_again(
        self, position
    ):
        position.add_to_position(A_HUNDRED_MILLION_UNITS_IN_STEPS, PRICE)
        position.add_to_position(ONE_STEP, PRICE)

        position.reduce_position(ONE_STEP)

        assert position.quantity_steps == A_HUNDRED_MILLION_UNITS_IN_STEPS

    def test_the_snapshot_reports_the_quantity_as_a_tradable_float(self, position):
        position.add_to_position(FIRST_FILL_STEPS, PRICE)

        assert position.snapshot().quantity == 0.029985
