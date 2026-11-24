import math

import numpy as np
import pytest

from robottraderslab.indicators import donchian_midline


@pytest.fixture
def high() -> np.ndarray:
    return np.array([10.0, 12.0, 11.0, 15.0, 13.0])


@pytest.fixture
def low() -> np.ndarray:
    return np.array([8.0, 9.0, 7.0, 10.0, 9.0])


class TestDonchianMidline:
    def test_period_three_averages_rolling_extremes(self, high, low):
        midline = donchian_midline(high, low, 3)

        assert math.isnan(midline[0])
        assert math.isnan(midline[1])
        assert midline[2] == pytest.approx((12.0 + 7.0) / 2)
        assert midline[3] == pytest.approx((15.0 + 7.0) / 2)
        assert midline[4] == pytest.approx((15.0 + 7.0) / 2)
