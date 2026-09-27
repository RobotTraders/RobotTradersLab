import pytest

from robottraderslab import _core, bootstrap, live

GATES = [_core, bootstrap, live]


@pytest.mark.parametrize("gate", GATES, ids=lambda gate: gate.__name__)
def test_every_listed_name_resolves(gate):
    missing = [name for name in gate.__all__ if not hasattr(gate, name)]

    assert not missing
