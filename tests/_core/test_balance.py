import pytest

from robottraderslab.exchanges import Balance


def test_init():
    balance = Balance(locked=4.3, total=7.7)

    assert balance.available == pytest.approx(3.4)
    assert balance.locked == 4.3
    assert balance.total == 7.7


class TestCompute:
    def test_available_is_computed(self):
        balance = Balance.compute(locked=11.0, total=15.0)
        assert balance.available == 4.0

    def test_locked_is_computed(self):
        balance = Balance.compute(available=4.0, total=15.0)
        assert balance.locked == 11.0

    def test_total_is_computed(self):
        balance = Balance.compute(available=4.0, locked=11.0)
        assert balance.total == 15.0

    def test_not_enough_args_provided(self):
        with pytest.raises(RuntimeError, match="Provide at least two"):
            Balance.compute()
