import pytest

from robottraderslab.bootstrap import load_simulated_market
from robottraderslab.exceptions import BotCriticalError


def test_uninstalled_market_type_raises_bot_critical_error():
    with pytest.raises(BotCriticalError):
        load_simulated_market("not_a_real_market_type")


def test_uninstalled_market_type_error_names_the_market_type():
    with pytest.raises(BotCriticalError, match="not_a_real_market_type"):
        load_simulated_market("not_a_real_market_type")
