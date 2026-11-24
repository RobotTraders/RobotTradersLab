import pytest

from robottraderslab_core.exceptions import (
    BotCriticalError,
    ExchangeCriticalError,
    ProfileConfigError,
    ProfileError,
)


@pytest.mark.parametrize("critical", [ExchangeCriticalError, BotCriticalError])
def test_a_critical_error_stands_outside_the_exception_tree(critical):
    assert issubclass(critical, BaseException)
    assert not issubclass(critical, Exception)


def test_a_refused_profile_is_a_profile_error():
    assert issubclass(ProfileConfigError, ProfileError)
