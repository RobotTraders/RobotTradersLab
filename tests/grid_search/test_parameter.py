import pytest

from robottraderslab.grid_search.parameter import (
    ExplicitParameter,
    RangeParameter,
    SamplingType,
)


class TestParameterLinearSampling:
    """Test linear sampling strategy."""

    def test_linear_sampling_generates_correct_values(self):
        param = RangeParameter(
            name="test_param",
            value_type=float,
            sampling=SamplingType.LINEAR,
            min_value=0.0,
            max_value=10.0,
            num_samples=5,
        )

        values = param.generate_values()

        assert len(values) == 5
        assert values == pytest.approx([0.0, 2.5, 5.0, 7.5, 10.0])

    def test_linear_sampling_validates_min_less_than_max(self):
        with pytest.raises(ValueError, match="min_value must be < max_value"):
            RangeParameter(
                name="test_param",
                value_type=float,
                sampling=SamplingType.LINEAR,
                min_value=10.0,
                max_value=5.0,
                num_samples=5,
            )

    def test_linear_sampling_requires_at_least_two_samples(self):
        with pytest.raises(ValueError, match="num_samples must be >= 2"):
            RangeParameter(
                name="test_param",
                value_type=float,
                sampling=SamplingType.LINEAR,
                min_value=0.0,
                max_value=10.0,
                num_samples=1,
            )


class TestRangeParameterValidation:
    def test_with_explicit_sampling_type(self):
        with pytest.raises(
            ValueError, match="RangeParameter requires LINEAR or LOG sampling"
        ):
            RangeParameter(
                name="test_param",
                value_type=float,
                sampling=SamplingType.EXPLICIT,
                min_value=1.0,
                max_value=10.0,
                num_samples=3,
            )


class TestParameterLogSampling:
    """Test logarithmic sampling strategy."""

    def test_log_sampling_generates_correct_values(self):
        param = RangeParameter(
            name="test_param",
            value_type=float,
            sampling=SamplingType.LOG,
            min_value=1.0,
            max_value=1000.0,
            num_samples=4,
        )

        values = param.generate_values()

        assert len(values) == 4
        assert values == pytest.approx([1.0, 10.0, 100.0, 1000.0])

    def test_log_sampling_requires_positive_min_value(self):
        with pytest.raises(ValueError, match="log sampling requires min_value > 0"):
            RangeParameter(
                name="test_param",
                value_type=float,
                sampling=SamplingType.LOG,
                min_value=0.0,
                max_value=100.0,
                num_samples=5,
            )


class TestParameterStepSampling:
    def test_step_generates_correct_values(self):
        param = RangeParameter(
            name="p",
            value_type=float,
            sampling=SamplingType.LINEAR,
            min_value=5.0,
            max_value=15.0,
            step=2.0,
        )

        values = param.generate_values()

        assert values == pytest.approx([5.0, 7.0, 9.0, 11.0, 13.0, 15.0])

    def test_step_always_includes_boundaries(self):
        param = RangeParameter(
            name="p",
            value_type=float,
            sampling=SamplingType.LINEAR,
            min_value=5.0,
            max_value=15.0,
            step=3.0,
        )

        values = param.generate_values()

        assert values == pytest.approx([5.0, 8.0, 11.0, 14.0, 15.0])

    def test_step_with_fractional_values(self):
        param = RangeParameter(
            name="p",
            value_type=float,
            sampling=SamplingType.LINEAR,
            min_value=0.0,
            max_value=1.0,
            step=0.3,
        )

        values = param.generate_values()

        assert values == pytest.approx([0.0, 0.3, 0.6, 0.9, 1.0])

    def test_step_with_an_integer_setting(self):
        param = RangeParameter(
            name="length",
            sampling=SamplingType.LINEAR,
            min_value=5.0,
            max_value=15.0,
            step=3.0,
            value_type=int,
        )

        values = param.generate_values()

        assert values == [5, 8, 11, 14, 15]
        assert all(isinstance(v, int) for v in values)

    def test_step_requires_linear_sampling(self):
        with pytest.raises(ValueError, match="step is only supported with linear"):
            RangeParameter(
                name="p",
                value_type=float,
                sampling=SamplingType.LOG,
                min_value=1.0,
                max_value=100.0,
                step=10.0,
            )

    def test_step_must_be_positive(self):
        with pytest.raises(ValueError, match="step must be > 0"):
            RangeParameter(
                name="p",
                value_type=float,
                sampling=SamplingType.LINEAR,
                min_value=0.0,
                max_value=10.0,
                step=-1.0,
            )

    def test_step_and_num_samples_mutually_exclusive(self):
        with pytest.raises(ValueError, match="exactly one of step or num_samples"):
            RangeParameter(
                name="p",
                value_type=float,
                sampling=SamplingType.LINEAR,
                min_value=0.0,
                max_value=10.0,
                num_samples=5,
                step=2.0,
            )

    def test_neither_step_nor_num_samples_fails(self):
        with pytest.raises(ValueError, match="exactly one of step or num_samples"):
            RangeParameter(
                name="p",
                value_type=float,
                sampling=SamplingType.LINEAR,
                min_value=0.0,
                max_value=10.0,
            )


class TestParameterExplicitSampling:
    """Test explicit value list sampling strategy."""

    def test_explicit_sampling_returns_provided_values(self):
        param = ExplicitParameter(
            name="test_param",
            value_type=float,
            values=(1.0, 5.0, 10.0, 50.0, 100.0),
        )

        values = param.generate_values()

        assert values == [1.0, 5.0, 10.0, 50.0, 100.0]

    def test_explicit_sampling_requires_at_least_two_values(self):
        with pytest.raises(ValueError, match="requires at least 2 values"):
            ExplicitParameter(
                name="test_param",
                value_type=float,
                values=(1.0,),
            )


class TestRangeParameterValueType:
    def test_a_float_setting_samples_floats(self):
        param = RangeParameter(
            name="p",
            value_type=float,
            sampling=SamplingType.LINEAR,
            min_value=0.0,
            max_value=10.0,
            num_samples=3,
        )

        values = param.generate_values()

        assert all(isinstance(v, float) for v in values)

    def test_an_integer_setting_samples_whole_numbers_linearly(self):
        param = RangeParameter(
            name="length",
            sampling=SamplingType.LINEAR,
            min_value=3.0,
            max_value=7.0,
            num_samples=3,
            value_type=int,
        )

        values = param.generate_values()

        assert values == [3, 5, 7]
        assert all(isinstance(v, int) for v in values)

    def test_an_integer_setting_samples_whole_numbers_logarithmically(self):
        param = RangeParameter(
            name="length",
            sampling=SamplingType.LOG,
            min_value=1.0,
            max_value=100.0,
            num_samples=3,
            value_type=int,
        )

        values = param.generate_values()

        assert values == [1, 10, 100]
        assert all(isinstance(v, int) for v in values)

    def test_an_integer_setting_rounds_to_the_nearest_whole(self):
        param = RangeParameter(
            name="window",
            sampling=SamplingType.LINEAR,
            min_value=1.0,
            max_value=4.0,
            num_samples=4,
            value_type=int,
        )

        values = param.generate_values()

        assert values == [1, 2, 3, 4]
        assert all(isinstance(v, int) for v in values)

    def test_a_text_setting_cannot_be_ranged(self):
        with pytest.raises(ValueError, match="integer or a float"):
            RangeParameter(
                name="p",
                sampling=SamplingType.LINEAR,
                min_value=0.0,
                max_value=10.0,
                num_samples=3,
                value_type=str,
            )


class TestExplicitParameterValueType:
    def test_a_float_setting_casts_values_to_floats(self):
        param = ExplicitParameter(name="p", values=(1, 2, 3), value_type=float)

        values = param.generate_values()

        assert all(isinstance(v, float) for v in values)

    def test_an_integer_setting_casts_values_to_whole_numbers(self):
        param = ExplicitParameter(
            name="length",
            values=(3.0, 5.0, 7.0),
            value_type=int,
        )

        values = param.generate_values()

        assert values == [3, 5, 7]
        assert all(isinstance(v, int) for v in values)

    def test_a_text_setting_keeps_text(self):
        param = ExplicitParameter(
            name="mode",
            values=("fast", "slow"),
            value_type=str,
        )

        values = param.generate_values()

        assert values == ["fast", "slow"]
        assert all(isinstance(v, str) for v in values)
