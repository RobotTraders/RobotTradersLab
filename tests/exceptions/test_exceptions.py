import pytest

from robottraderslab.exceptions import ExchangeCriticalError, StrategyCriticalError


@pytest.mark.parametrize("critical", [ExchangeCriticalError, StrategyCriticalError])
def test_a_critical_error_stands_outside_the_exception_tree(critical):
    assert issubclass(critical, BaseException)
    assert not issubclass(critical, Exception)
