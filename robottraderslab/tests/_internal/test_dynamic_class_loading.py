from collections import Counter
from collections.abc import Callable
from importlib.metadata import EntryPoint
from unittest.mock import Mock

import pytest

from robottraderslab._internal.dynamic_class_loading import (
    ClassLoadingError,
    find_class,
    load_class,
    registered_entry_point_name,
)

_GROUP = "robot_traders_lab.test_group"
_ENTRY_POINT = EntryPoint(name="counter", value="collections:Counter", group=_GROUP)


@pytest.fixture
def installed_entry_points(
    install_entry_points: Callable[[list[EntryPoint]], Mock],
) -> Mock:
    return install_entry_points([_ENTRY_POINT])


class TestLoadClassByEntryPointName:
    def test_resolves_the_installed_entry_point(self, installed_entry_points):
        assert load_class("counter", _GROUP) is Counter

    def test_name_matching_ignores_case(self, installed_entry_points):
        assert load_class("COUNTER", _GROUP) is Counter

    def test_unknown_name(self, installed_entry_points):
        with pytest.raises(ClassLoadingError, match="missing"):
            load_class("missing", _GROUP)

    def test_installed_distributions_are_read_once_per_name(
        self, installed_entry_points
    ):
        load_class("counter", _GROUP)
        load_class("counter", _GROUP)

        installed_entry_points.assert_called_once_with(group=_GROUP)

    def test_a_failed_lookup_is_retried(self, installed_entry_points):
        with pytest.raises(ClassLoadingError):
            load_class("missing", _GROUP)
        with pytest.raises(ClassLoadingError):
            load_class("missing", _GROUP)

        assert installed_entry_points.call_count == 2


class TestFindClass:
    def test_a_registered_name(self, installed_entry_points):
        assert find_class("Counter", _GROUP) is Counter

    def test_a_name_no_distribution_registers(self, installed_entry_points):
        assert find_class("missing", _GROUP) is None


class TestRegisteredEntryPointName:
    def test_a_name_spelt_in_another_case(self, installed_entry_points):
        assert registered_entry_point_name("COUNTER", _GROUP) == "counter"

    def test_a_path_written_with_a_colon(self, installed_entry_points):
        assert registered_entry_point_name("collections:Counter", _GROUP) == "counter"

    def test_a_path_written_with_dots(self, installed_entry_points):
        assert registered_entry_point_name("collections.Counter", _GROUP) == "counter"

    def test_a_name_no_distribution_registers(self, installed_entry_points):
        assert registered_entry_point_name("missing", _GROUP) is None

    def test_a_path_no_distribution_registers(self, installed_entry_points):
        assert registered_entry_point_name("collections.OrderedDict", _GROUP) is None
