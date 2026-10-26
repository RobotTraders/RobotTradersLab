import pytest

from robottraderslab.exceptions import BotCriticalError, ExchangeCriticalError


@pytest.mark.parametrize("critical", [ExchangeCriticalError, BotCriticalError])
def test_a_critical_error_stands_outside_the_exception_tree(critical):
    assert issubclass(critical, BaseException)
    assert not issubclass(critical, Exception)
