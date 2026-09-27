import pytest

from robottraderslab.bootstrap import load_simulated_market
from robottraderslab.exceptions import StrategyCriticalError


def test_uninstalled_market_type_raises_strategy_critical_error():
    with pytest.raises(StrategyCriticalError):
        load_simulated_market("not_a_real_market_type")


def test_uninstalled_market_type_error_names_the_market_type():
    with pytest.raises(StrategyCriticalError, match="not_a_real_market_type"):
        load_simulated_market("not_a_real_market_type")
