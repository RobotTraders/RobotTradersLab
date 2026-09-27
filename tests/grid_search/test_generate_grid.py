import pytest

from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.grid_search.optimiser import _generate_grid
from robottraderslab.grid_search.parameter import (
    ExplicitParameter,
    RangeParameter,
    SamplingType,
)


class TestGridGeneratorValidation:
    """Test GridGenerator validation logic."""

    def test_rejects_single_parameter(self):
        params = [
            RangeParameter(
                name="param1",
                value_type=float,
                sampling=SamplingType.LINEAR,
                min_value=0.0,
                max_value=10.0,
                num_samples=5,
            )
        ]

        with pytest.raises(
            StrategyCriticalError,
            match=r"`\[optimisation\]` declares 1\.",
        ):
            _generate_grid(params)

    def test_rejects_three_parameters(self):
        params = [
            RangeParameter(
                name="param1",
                value_type=float,
                sampling=SamplingType.LINEAR,
                min_value=0.0,
                max_value=10.0,
                num_samples=5,
            ),
            RangeParameter(
                name="param2",
                value_type=float,
                sampling=SamplingType.LINEAR,
                min_value=0.0,
                max_value=10.0,
                num_samples=5,
            ),
            RangeParameter(
                name="param3",
                value_type=float,
                sampling=SamplingType.LINEAR,
                min_value=0.0,
                max_value=10.0,
                num_samples=5,
            ),
        ]

        with pytest.raises(
            StrategyCriticalError,
            match=r"`\[optimisation\]` declares 3\..*`\[strategy\]`",
        ):
            _generate_grid(params)

    def test_too_few_parameters_has_none_left_to_place(self):
        params = [
            RangeParameter(
                name="param1",
                value_type=float,
                sampling=SamplingType.LINEAR,
                min_value=0.0,
                max_value=10.0,
                num_samples=5,
            )
        ]

        with pytest.raises(StrategyCriticalError) as refusal:
            _generate_grid(params)

        assert "`[strategy]`" not in str(refusal.value)


class TestGridGeneration:
    """Test grid generation algorithm."""

    def test_generates_all_combinations_linear(self):
        params = [
            RangeParameter(
                name="trix_length",
                value_type=float,
                sampling=SamplingType.LINEAR,
                min_value=5.0,
                max_value=10.0,
                num_samples=3,
            ),
            RangeParameter(
                name="signal_length",
                value_type=float,
                sampling=SamplingType.LINEAR,
                min_value=10.0,
                max_value=20.0,
                num_samples=2,
            ),
        ]

        grid_points = _generate_grid(params)

        assert len(grid_points) == 6
        assert grid_points[0].parameters == {
            "trix_length": 5.0,
            "signal_length": 10.0,
        }
        assert grid_points[-1].parameters == {
            "trix_length": 10.0,
            "signal_length": 20.0,
        }

    def test_generates_all_combinations_mixed_sampling(self):
        params = [
            ExplicitParameter(
                name="param1",
                value_type=float,
                values=(1.0, 2.0, 3.0),
            ),
            RangeParameter(
                name="param2",
                value_type=float,
                sampling=SamplingType.LOG,
                min_value=1.0,
                max_value=100.0,
                num_samples=3,
            ),
        ]

        grid_points = _generate_grid(params)

        assert len(grid_points) == 9

    def test_grid_point_order_is_deterministic(self):
        params = [
            ExplicitParameter(
                name="param1",
                value_type=float,
                values=(1.0, 2.0),
            ),
            ExplicitParameter(
                name="param2",
                value_type=float,
                values=(10.0, 20.0),
            ),
        ]

        grid_points = _generate_grid(params)

        assert grid_points[0].parameters == {"param1": 1.0, "param2": 10.0}
        assert grid_points[1].parameters == {"param1": 1.0, "param2": 20.0}
        assert grid_points[2].parameters == {"param1": 2.0, "param2": 10.0}
        assert grid_points[3].parameters == {"param1": 2.0, "param2": 20.0}

    def test_grid_keeps_whole_numbers_for_integer_settings(self):
        params = [
            RangeParameter(
                name="fast_ma",
                sampling=SamplingType.LINEAR,
                min_value=3.0,
                max_value=7.0,
                num_samples=3,
                value_type=int,
            ),
            RangeParameter(
                name="slow_ma",
                sampling=SamplingType.LINEAR,
                min_value=10.0,
                max_value=20.0,
                num_samples=2,
                value_type=int,
            ),
        ]

        grid_points = _generate_grid(params)

        for point in grid_points:
            assert isinstance(point.parameters["fast_ma"], int)
            assert isinstance(point.parameters["slow_ma"], int)
