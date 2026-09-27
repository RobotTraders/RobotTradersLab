import pytest

from robottraderslab._core import DrawdownUnit
from robottraderslab.plotting.plot_base import drawdown_axis_label


@pytest.mark.parametrize(
    ("unit", "label"),
    [
        (DrawdownUnit.PERCENT, "Drawdown %"),
        (DrawdownUnit.CURRENCY, "Drawdown in Quote"),
    ],
)
def test_the_drawdown_axis_is_named_by_its_unit(unit, label):
    assert drawdown_axis_label(unit) == label
