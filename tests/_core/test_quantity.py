import pytest

from robottraderslab._core import to_quantity, to_step_count


class TestConversionBoundary:
    @pytest.mark.parametrize("steps", [1, 2_998_500, 4_997_500, 10_000_000_000_000])
    def test_a_reported_quantity_comes_back_as_the_same_count(self, steps):
        assert to_step_count(to_quantity(steps)) == steps

    @pytest.mark.parametrize(
        ("quantity", "expected_steps"),
        [(0.000000014, 1), (0.000000019, 2), (0.0299850004, 2_998_500)],
    )
    def test_a_quantity_between_steps_takes_the_nearest_step(
        self, quantity, expected_steps
    ):
        assert to_step_count(quantity) == expected_steps

    def test_one_step_is_a_hundred_millionth_of_a_unit(self):
        assert to_quantity(1) == 1e-8
