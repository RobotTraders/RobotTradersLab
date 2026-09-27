import pytest

from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.grid_search.optimisation_config import OptimisationConfig
from robottraderslab.grid_search.parameter import SamplingType

CURRENT_VALUES = {
    "strategy.profiles.param1": 0.0,
    "strategy.profiles.param2": 0.0,
    "strategy.profiles.p1": 0.0,
    "strategy.profiles.p2": 0.0,
}


class TestOptimisationConfig:
    """Test OptimisationConfig model methods."""

    def test_from_config(self):
        config_dict = {
            "parameter_configs": [
                {
                    "parameter": "strategy.profiles.trix_length",
                    "sampling": "linear",
                    "min": 5.0,
                    "max": 15.0,
                    "num_samples": 6,
                },
                {
                    "parameter": "strategy.profiles.signal_length",
                    "sampling": "log",
                    "min": 10.0,
                    "max": 100.0,
                    "num_samples": 4,
                },
            ],
        }

        config = OptimisationConfig.from_config(config_dict)

        assert len(config.parameter_configs) == 2
        assert config.parameter_configs[0].parameter == "strategy.profiles.trix_length"
        assert config.parameter_configs[0].sampling == SamplingType.LINEAR
        assert (
            config.parameter_configs[1].parameter == "strategy.profiles.signal_length"
        )
        assert config.parameter_configs[1].sampling == SamplingType.LOG

    def test_to_parameters(self):
        config_dict = {
            "parameter_configs": [
                {
                    "parameter": "strategy.profiles.param1",
                    "sampling": "linear",
                    "min": 0.0,
                    "max": 10.0,
                    "num_samples": 5,
                },
                {
                    "parameter": "strategy.profiles.param2",
                    "sampling": "explicit",
                    "values": [1.0, 5.0, 10.0],
                },
            ],
        }

        config = OptimisationConfig.from_config(config_dict)
        parameters = config.to_parameters(CURRENT_VALUES)

        assert len(parameters) == 2
        assert parameters[0].name == "strategy.profiles.param1"
        assert parameters[0].sampling == SamplingType.LINEAR
        assert parameters[0].min_value == 0.0
        assert parameters[0].max_value == 10.0
        assert parameters[0].num_samples == 5
        assert parameters[1].name == "strategy.profiles.param2"
        assert parameters[1].values == (1.0, 5.0, 10.0)

    def test_to_parameters_explicit_without_values(self):
        config = OptimisationConfig.from_config(
            {
                "parameter_configs": [
                    {
                        "parameter": "strategy.profiles.param1",
                        "sampling": "explicit",
                    },
                ],
            }
        )

        with pytest.raises(
            StrategyCriticalError, match="explicit sampling requires values"
        ):
            config.to_parameters(CURRENT_VALUES)

    def test_to_parameters_range_without_required_fields(self):
        config = OptimisationConfig.from_config(
            {
                "parameter_configs": [
                    {
                        "parameter": "strategy.profiles.param1",
                        "sampling": "linear",
                        "min": 0.0,
                    },
                ],
            }
        )

        with pytest.raises(
            StrategyCriticalError, match="linear sampling requires min and max"
        ):
            config.to_parameters(CURRENT_VALUES)

    def test_to_parameters_range_without_num_samples_or_step(self):
        config = OptimisationConfig.from_config(
            {
                "parameter_configs": [
                    {
                        "parameter": "strategy.profiles.param1",
                        "sampling": "linear",
                        "min": 0.0,
                        "max": 10.0,
                    },
                ],
            }
        )

        with pytest.raises(
            StrategyCriticalError, match="linear sampling requires num_samples or step"
        ):
            config.to_parameters(CURRENT_VALUES)

    def test_to_parameters_with_step(self):
        config = OptimisationConfig.from_config(
            {
                "parameter_configs": [
                    {
                        "parameter": "strategy.profiles.p1",
                        "sampling": "linear",
                        "min": 5.0,
                        "max": 15.0,
                        "step": 2.0,
                    },
                    {
                        "parameter": "strategy.profiles.p2",
                        "sampling": "linear",
                        "min": 0.0,
                        "max": 10.0,
                        "num_samples": 3,
                    },
                ],
            }
        )

        parameters = config.to_parameters(CURRENT_VALUES)

        assert parameters[0].step == 2.0
        assert parameters[0].num_samples is None
        assert parameters[1].num_samples == 3
        assert parameters[1].step is None


class TestOptimisationConfigValueType:
    def test_a_dtype_declaration_is_refused(self):
        with pytest.raises(StrategyCriticalError, match="dtype"):
            OptimisationConfig.from_config(
                {
                    "parameter_configs": [
                        {
                            "parameter": "strategy.profiles.length",
                            "sampling": "linear",
                            "dtype": "int",
                            "min": 3.0,
                            "max": 7.0,
                            "num_samples": 3,
                        },
                    ],
                }
            )

    def test_a_range_takes_the_type_of_the_setting_it_varies(self):
        config = OptimisationConfig.from_config(
            {
                "parameter_configs": [
                    {
                        "parameter": "strategy.profiles.fast",
                        "sampling": "linear",
                        "min": 3.0,
                        "max": 7.0,
                        "num_samples": 3,
                    },
                    {
                        "parameter": "strategy.profiles.slow",
                        "sampling": "linear",
                        "min": 10.0,
                        "max": 20.0,
                        "num_samples": 3,
                    },
                ],
            }
        )

        parameters = config.to_parameters(
            {"strategy.profiles.fast": 5, "strategy.profiles.slow": 12.5}
        )

        assert parameters[0].generate_values() == [3, 5, 7]
        assert parameters[1].generate_values() == [10.0, 15.0, 20.0]

    @pytest.mark.parametrize(
        ("path", "current_value"),
        [
            ("strategy.profiles.mode", "fast"),
            ("strategy.profiles.long_only", True),
        ],
    )
    def test_a_range_over_a_setting_that_is_no_number_is_refused(
        self, path, current_value
    ):
        config = OptimisationConfig.from_config(
            {
                "parameter_configs": [
                    {
                        "parameter": path,
                        "sampling": "linear",
                        "min": 0.0,
                        "max": 1.0,
                        "num_samples": 2,
                    },
                ],
            }
        )

        with pytest.raises(ValueError, match="integer or a float"):
            config.to_parameters({path: current_value})

    def test_explicit_values_take_the_type_of_the_setting_they_vary(self):
        config = OptimisationConfig.from_config(
            {
                "parameter_configs": [
                    {
                        "parameter": "strategy.profiles.length",
                        "sampling": "explicit",
                        "values": [3.0, 5.0, 7.0],
                    },
                    {
                        "parameter": "strategy.profiles.mode",
                        "sampling": "explicit",
                        "values": ["fast", "slow"],
                    },
                ],
            }
        )

        parameters = config.to_parameters(
            {"strategy.profiles.length": 8, "strategy.profiles.mode": "fast"}
        )

        assert parameters[0].generate_values() == [3, 5, 7]
        assert parameters[1].generate_values() == ["fast", "slow"]
