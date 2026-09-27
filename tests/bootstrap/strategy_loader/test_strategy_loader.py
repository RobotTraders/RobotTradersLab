from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from robottraderslab.bootstrap import load_lightweight_chart_indicators, load_strategy
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.strategies import Candles, StrategyProtocol
from strategy_loader.strategy_loader_stubs import StrategyCapturesSettings


def _mock_resolve_entry_point(name: str, group: str) -> str:
    """Answer for the installed strategies, so a test can name one that is not."""
    test_mappings = {
        "robot_traders_lab.strategies": {
            "captures_settings": "strategy_loader.strategy_loader_stubs:StrategyCapturesSettings",
            "implements_protocol": "strategy_loader.strategy_loader_stubs:StrategyImplementsProtocol",
            "subclasses_protocol": "strategy_loader.strategy_loader_stubs:StrategySubclassesProtocol",
            "doesnt_follow_protocol": "strategy_loader.strategy_loader_stubs:StrategyDoesntFollowProtocol",
        },
        "robot_traders_lab.indicators": {
            "mock_indicators": "robottraderslab.ma_strategy.futures_ma_strategy:get_lightweight_chart_indicators",
        },
    }
    group_mappings = test_mappings.get(group, {})
    if name in group_mappings:
        return group_mappings[name]
    from robottraderslab._core.dynamic_class_loading import ClassLoadingError

    raise ClassLoadingError(f"'{name}' not found in entry point group '{group}'")


@pytest.fixture(autouse=True)
def _mock_entry_points() -> None:
    with patch(
        "robottraderslab._core.dynamic_class_loading._resolve_entry_point",
        side_effect=_mock_resolve_entry_point,
    ):
        yield


class TestLoadStrategy:
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

        with pytest.raises(
            StrategyCriticalError, match="Module `a.b.c` can't be imported"
        ):
            load_strategy(settings)

    def test_nonexistent_class(self):
        settings = {
            "strategy_class": "strategy_loader.strategy_loader_stubs.NonExistentClass"
        }

        with pytest.raises(
            StrategyCriticalError,
            match="There is no `NonExistentClass` class in module `strategy_loader.strategy_loader_stubs`",
        ):
            load_strategy(settings)

    def test_missing_strategy_class_setting(self):
        settings = {}

        with pytest.raises(
            StrategyCriticalError, match="`strategy_class` is a required config key"
        ):
            load_strategy(settings)

    def test_non_dotted_class_path_not_in_entry_points(self):
        settings = {"strategy_class": "NotRegistered"}

        with pytest.raises(
            StrategyCriticalError,
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

    def test_successful_load_via_dotted_path(self, candles):
        lines = load_lightweight_chart_indicators(
            "robottraderslab.ma_strategy.futures_ma_strategy.get_lightweight_chart_indicators",
            candles,
            {"fast_ma_length": 10, "slow_ma_length": 30},
        )

        assert {line.name for line in lines} == {"SMA 10", "SMA 30"}

    def test_not_in_entry_points(self, candles):
        with pytest.raises(
            StrategyCriticalError,
            match=r"'nonexistent' not found in entry point group",
        ):
            load_lightweight_chart_indicators(
                "nonexistent", candles, {"fast_ma_length": 5}
            )

    def test_nonexistent_module_via_dotted_path(self, candles):
        with pytest.raises(
            StrategyCriticalError,
            match=r"Module `a\.b\.c` can't be imported",
        ):
            load_lightweight_chart_indicators(
                "a.b.c.get_indicators", candles, {"fast_ma_length": 5}
            )

    def test_nonexistent_function_in_module(self, candles):
        with pytest.raises(
            StrategyCriticalError,
            match=r"There is no `nonexistent_func` class in module",
        ):
            load_lightweight_chart_indicators(
                "robottraderslab.ma_strategy.futures_ma_strategy.nonexistent_func",
                candles,
                {"fast_ma_length": 5},
            )
