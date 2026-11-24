import re
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from robottraderslab.bootstrap import load_lightweight_chart_indicators, load_strategy
from robottraderslab.exceptions import BotCriticalError
from robottraderslab.strategies import Candles, StrategyProtocol
from strategy_loader.strategy_loader_stubs import StrategyCapturesSettings

_STRATEGIES_GROUP = "robot_traders_lab.strategies"
_INDICATORS_GROUP = "robot_traders_lab.indicators"
_STUBS = "strategy_loader.strategy_loader_stubs"

_INSTALLED_ENTRY_POINTS = [
    EntryPoint(
        "captures_settings", f"{_STUBS}:StrategyCapturesSettings", _STRATEGIES_GROUP
    ),
    EntryPoint(
        "implements_protocol", f"{_STUBS}:StrategyImplementsProtocol", _STRATEGIES_GROUP
    ),
    EntryPoint(
        "subclasses_protocol", f"{_STUBS}:StrategySubclassesProtocol", _STRATEGIES_GROUP
    ),
    EntryPoint(
        "doesnt_follow_protocol",
        f"{_STUBS}:StrategyDoesntFollowProtocol",
        _STRATEGIES_GROUP,
    ),
    EntryPoint(
        "mock_indicators",
        "robottraderslab.ma_strategy.futures_ma_strategy:get_lightweight_chart_indicators",
        _INDICATORS_GROUP,
    ),
    EntryPoint(
        "raising_indicators", f"{_STUBS}:raising_chart_lines", _INDICATORS_GROUP
    ),
    EntryPoint("unloadable_indicators", "a.b.c:chart_lines", _INDICATORS_GROUP),
]


@pytest.fixture(autouse=True)
def _installed_entry_points(install_entry_points) -> None:
    install_entry_points(_INSTALLED_ENTRY_POINTS)


class _RefusingStrategy:
    def __init__(self, **settings):
        raise BotCriticalError("profile 1 (BTC/USDT:USDT): tag: 'tp' is reserved")


class TestLoadStrategy:
    def test_a_refusal_names_the_file_the_sections_were_read_from(self):
        settings = {"strategy_class": "refusing"}
        config_file = Path("bots") / "mine.toml"

        with patch(
            "robottraderslab.bootstrap.strategy_loader._load_strategy_class",
            return_value=_RefusingStrategy,
        ):
            with pytest.raises(
                BotCriticalError,
                match=f"`{re.escape(str(config_file))}`: profile 1",
            ):
                load_strategy(settings, config_file=config_file)

    def test_a_refusal_with_no_file_given_is_left_as_it_is(self):
        settings = {"strategy_class": "refusing"}

        with patch(
            "robottraderslab.bootstrap.strategy_loader._load_strategy_class",
            return_value=_RefusingStrategy,
        ):
            with pytest.raises(BotCriticalError, match=r"^profile 1"):
                load_strategy(settings)

    @pytest.mark.parametrize(
        "strategy_class",
        [
            "implements_protocol",
            "subclasses_protocol",
            "strategy_loader.strategy_loader_stubs.StrategyImplementsProtocol",
            "strategy_loader.strategy_loader_stubs.StrategySubclassesProtocol",
        ],
    )
    def test_valid_strategy(self, strategy_class):
        settings = {"strategy_class": strategy_class}

        strategy = load_strategy(settings)

        assert isinstance(strategy, StrategyProtocol)

    def test_strategy_class_setting_removed(self):
        settings = {"strategy_class": "captures_settings"}

        strategy = cast(StrategyCapturesSettings, load_strategy(settings))

        assert "strategy_class" not in strategy.received_settings


class TestLoadStrategyErrors:
    def test_invalid_protocol_implementation(self):
        settings = {"strategy_class": "doesnt_follow_protocol"}

        with pytest.raises(
            TypeError,
            match="Make sure your strategy class.*doesnt_follow_protocol.*implements `StrategyProtocol`",
        ):
            load_strategy(settings)

    def test_nonexistent_module(self):
        settings = {"strategy_class": "a.b.c.SomeStrategy"}

        with pytest.raises(BotCriticalError, match="Module `a.b.c` can't be imported"):
            load_strategy(settings)

    def test_nonexistent_class(self):
        settings = {
            "strategy_class": "strategy_loader.strategy_loader_stubs.NonExistentClass"
        }

        with pytest.raises(
            BotCriticalError,
            match="There is no `NonExistentClass` class in module `strategy_loader.strategy_loader_stubs`",
        ):
            load_strategy(settings)

    def test_missing_strategy_class_setting(self):
        settings = {}

        with pytest.raises(
            BotCriticalError, match="`strategy_class` is a required config key"
        ):
            load_strategy(settings)

    def test_non_dotted_class_path_not_in_entry_points(self):
        settings = {"strategy_class": "NotRegistered"}

        with pytest.raises(
            BotCriticalError,
            match="'NotRegistered' not found in entry point",
        ):
            load_strategy(settings)


class TestLoadLightweightChartIndicators:
    @pytest.fixture
    def candles(self) -> Candles:
        closes = np.arange(100.0, 150.0)
        return Candles(
            open=closes,
            high=closes + 1.0,
            low=closes - 1.0,
            close=closes,
            volume=np.ones_like(closes),
        )

    def test_successful_load_via_entry_point(self, candles):
        lines = load_lightweight_chart_indicators(
            "mock_indicators", candles, {"fast_ma_length": 10, "slow_ma_length": 30}
        )

        by_name = {line.name: line for line in lines}
        assert by_name["SMA 10"].colour == "blue"
        assert by_name["SMA 30"].colour == "red"
        assert len(by_name["SMA 10"].values) == len(candles.close)

    def test_a_strategy_registering_no_indicators(self, candles):
        lines = load_lightweight_chart_indicators(
            "nonexistent", candles, {"fast_ma_length": 5}
        )

        assert lines == []

    def test_a_full_class_path_registering_no_indicators(self, candles):
        lines = load_lightweight_chart_indicators(
            "some_package.main.SomeStrategy", candles, {"fast_ma_length": 5}
        )

        assert lines == []

    def test_an_indicators_function_that_raises(self, candles):
        with pytest.raises(RuntimeError, match="indicator maths broke"):
            load_lightweight_chart_indicators(
                "raising_indicators", candles, {"fast_ma_length": 5}
            )

    def test_an_indicators_entry_point_whose_module_is_missing(self, candles):
        with pytest.raises(
            BotCriticalError, match=r"Module `a\.b\.c` can't be imported"
        ):
            load_lightweight_chart_indicators(
                "unloadable_indicators", candles, {"fast_ma_length": 5}
            )
