import pytest

from robottraderslab.exceptions import BotCriticalError
from robottraderslab.grid_search.optimisation_config import (
    OptimisationConfig,
    ProfileSelector,
)
from robottraderslab.grid_search.parameter import SamplingType

CURRENT_VALUES = {
    "param1": 0.0,
    "param2": 0.0,
    "p1": 0.0,
    "p2": 0.0,
}


class TestOptimisationConfig:
    """Test OptimisationConfig model methods."""

    def test_from_config(self):
        config_dict = {
            "profile": {"symbol": "BTC/USDT:USDT"},
            "parameter_configs": [
                {
                    "parameter": "trix_length",
                    "sampling": "linear",
                    "min": 5.0,
                    "max": 15.0,
                    "num_samples": 6,
                },
                {
                    "parameter": "signal_length",
                    "sampling": "log",
                    "min": 10.0,
                    "max": 100.0,
                    "num_samples": 4,
                },
            ],
        }

        config = OptimisationConfig.from_config(config_dict)

        assert len(config.parameter_configs) == 2
        assert config.parameter_configs[0].parameter == "trix_length"
        assert config.parameter_configs[0].sampling == SamplingType.LINEAR
        assert config.parameter_configs[1].parameter == "signal_length"
        assert config.parameter_configs[1].sampling == SamplingType.LOG

    def test_to_parameters(self):
        config_dict = {
            "profile": {"symbol": "BTC/USDT:USDT"},
            "parameter_configs": [
                {
                    "parameter": "param1",
                    "sampling": "linear",
                    "min": 0.0,
                    "max": 10.0,
                    "num_samples": 5,
                },
                {
                    "parameter": "param2",
                    "sampling": "explicit",
                    "values": [1.0, 5.0, 10.0],
                },
            ],
        }

        config = OptimisationConfig.from_config(config_dict)
        parameters = config.to_parameters(CURRENT_VALUES)

        assert len(parameters) == 2
        assert parameters[0].name == "param1"
        assert parameters[0].sampling == SamplingType.LINEAR
        assert parameters[0].min_value == 0.0
        assert parameters[0].max_value == 10.0
        assert parameters[0].num_samples == 5
        assert parameters[1].name == "param2"
        assert parameters[1].values == (1.0, 5.0, 10.0)

    def test_to_parameters_explicit_without_values(self):
        config = OptimisationConfig.from_config(
            {
                "profile": {"symbol": "BTC/USDT:USDT"},
                "parameter_configs": [
                    {
                        "parameter": "param1",
                        "sampling": "explicit",
                    },
                ],
            }
        )

        with pytest.raises(BotCriticalError, match="explicit sampling requires values"):
            config.to_parameters(CURRENT_VALUES)

    def test_to_parameters_range_without_required_fields(self):
        config = OptimisationConfig.from_config(
            {
                "profile": {"symbol": "BTC/USDT:USDT"},
                "parameter_configs": [
                    {
                        "parameter": "param1",
                        "sampling": "linear",
                        "min": 0.0,
                    },
                ],
            }
        )

        with pytest.raises(
            BotCriticalError, match="linear sampling requires min and max"
        ):
            config.to_parameters(CURRENT_VALUES)

    def test_to_parameters_range_without_num_samples_or_step(self):
        config = OptimisationConfig.from_config(
            {
                "profile": {"symbol": "BTC/USDT:USDT"},
                "parameter_configs": [
                    {
                        "parameter": "param1",
                        "sampling": "linear",
                        "min": 0.0,
                        "max": 10.0,
                    },
                ],
            }
        )

        with pytest.raises(
            BotCriticalError, match="linear sampling requires num_samples or step"
        ):
            config.to_parameters(CURRENT_VALUES)

    def test_to_parameters_with_step(self):
        config = OptimisationConfig.from_config(
            {
                "profile": {"symbol": "BTC/USDT:USDT"},
                "parameter_configs": [
                    {
                        "parameter": "p1",
                        "sampling": "linear",
                        "min": 5.0,
                        "max": 15.0,
                        "step": 2.0,
                    },
                    {
                        "parameter": "p2",
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
        with pytest.raises(BotCriticalError, match="dtype"):
            OptimisationConfig.from_config(
                {
                    "profile": {"symbol": "BTC/USDT:USDT"},
                    "parameter_configs": [
                        {
                            "parameter": "length",
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
                "profile": {"symbol": "BTC/USDT:USDT"},
                "parameter_configs": [
                    {
                        "parameter": "fast",
                        "sampling": "linear",
                        "min": 3.0,
                        "max": 7.0,
                        "num_samples": 3,
                    },
                    {
                        "parameter": "slow",
                        "sampling": "linear",
                        "min": 10.0,
                        "max": 20.0,
                        "num_samples": 3,
                    },
                ],
            }
        )

        parameters = config.to_parameters({"fast": 5, "slow": 12.5})

        assert parameters[0].generate_values() == [3, 5, 7]
        assert parameters[1].generate_values() == [10.0, 15.0, 20.0]

    @pytest.mark.parametrize(
        ("path", "current_value"),
        [
            ("mode", "fast"),
            ("long_only", True),
        ],
    )
    def test_a_range_over_a_setting_that_is_no_number_is_refused(
        self, path, current_value
    ):
        config = OptimisationConfig.from_config(
            {
                "profile": {"symbol": "BTC/USDT:USDT"},
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
                "profile": {"symbol": "BTC/USDT:USDT"},
                "parameter_configs": [
                    {
                        "parameter": "length",
                        "sampling": "explicit",
                        "values": [3.0, 5.0, 7.0],
                    },
                    {
                        "parameter": "mode",
                        "sampling": "explicit",
                        "values": ["fast", "slow"],
                    },
                ],
            }
        )

        parameters = config.to_parameters({"length": 8, "mode": "fast"})

        assert parameters[0].generate_values() == [3, 5, 7]
        assert parameters[1].generate_values() == ["fast", "slow"]


class TestFromFile:
    SEARCH = (
        "max_workers = 1\n\n"
        "[profile]\n"
        'symbol = "BTC/USDT:USDT"\n\n'
        "[[parameter_configs]]\n"
        'parameter = "length"\n'
        'sampling = "linear"\n'
        "min = 1.0\nmax = 2.0\nnum_samples = 2\n"
    )

    def test_reads_the_search_a_file_declares(self, tmp_path):
        search_file = tmp_path / "grid-search.toml"
        search_file.write_text(self.SEARCH, encoding="utf-8")

        config = OptimisationConfig.from_file(search_file)

        assert config.max_workers == 1
        assert config.parameter_configs[0].parameter == "length"

    @pytest.mark.parametrize("section", ["strategy", "backtest", "live"])
    def test_a_section_of_the_bot_is_refused_naming_bot_toml(self, tmp_path, section):
        search_file = tmp_path / "grid-search.toml"
        search_file.write_text(f"[{section}]\n" + self.SEARCH, encoding="utf-8")

        with pytest.raises(BotCriticalError, match="bot.toml") as refusal:
            OptimisationConfig.from_file(search_file)

        assert f"[{section}]" in str(refusal.value)

    def test_a_file_that_is_not_valid_toml_is_refused(self, tmp_path):
        search_file = tmp_path / "grid-search.toml"
        search_file.write_text("[[parameter_configs]\n", encoding="utf-8")

        with pytest.raises(BotCriticalError, match="is not valid TOML"):
            OptimisationConfig.from_file(search_file)

    def test_a_file_that_is_not_there_is_refused(self, tmp_path):
        with pytest.raises(BotCriticalError, match="no grid search file"):
            OptimisationConfig.from_file(tmp_path / "grid-search.toml")


class TestProfileSelector:
    PROFILES = (
        {"symbol": "ETH/USDT:USDT", "timeframe": "4h", "tag": "alpha"},
        {"symbol": "ETH/USDT:USDT", "timeframe": "4h", "tag": "beta"},
        {"symbol": "BTC/USDT:USDT", "timeframe": "1h", "tag": "alpha"},
    )

    def test_the_keys_given_pick_one_profile_wherever_it_stands(self):
        selector = ProfileSelector(symbol="ETH/USDT:USDT", timeframe="4h", tag="beta")

        assert selector.index_in(self.PROFILES) == 1
        assert selector.index_in(self.PROFILES[::-1]) == 1

    def test_a_profile_matching_none_is_refused_naming_the_keys_and_the_profiles(self):
        selector = ProfileSelector(symbol="SOL/USDT:USDT", timeframe="4h")

        with pytest.raises(BotCriticalError, match="matches no profile") as refusal:
            selector.index_in(self.PROFILES)

        message = str(refusal.value)
        assert "symbol = SOL/USDT:USDT, timeframe = 4h" in message
        assert "symbol = BTC/USDT:USDT, timeframe = 1h, tag = alpha" in message

    def test_a_profile_matching_several_is_refused_naming_the_keys_and_the_profiles(
        self,
    ):
        selector = ProfileSelector(symbol="ETH/USDT:USDT", timeframe="4h")

        with pytest.raises(BotCriticalError, match="matches more than one") as refusal:
            selector.index_in(self.PROFILES)

        assert "tag = beta" in str(refusal.value)

    def test_a_profile_giving_no_key_is_refused(self):
        with pytest.raises(BotCriticalError, match="none of"):
            OptimisationConfig.from_config(
                {
                    "profile": {},
                    "parameter_configs": [],
                }
            )

    def test_a_search_without_a_profile_is_refused(self):
        with pytest.raises(BotCriticalError, match="profile"):
            OptimisationConfig.from_config({"parameter_configs": []})

    def test_a_key_that_is_not_an_identity_is_refused(self):
        with pytest.raises(BotCriticalError, match="trix_length"):
            OptimisationConfig.from_config(
                {
                    "profile": {"symbol": "BTC/USDT:USDT", "trix_length": 8},
                    "parameter_configs": [],
                }
            )
