import math

import numpy as np
import pytest

from robottraderslab.indicators import MAType, ema, moving_average, sma, wma


@pytest.fixture
def five_closes() -> np.ndarray:
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def three_closes() -> np.ndarray:
    return np.array([1.0, 2.0, 3.0])


class TestSma:
    def test_period_three_yields_trailing_means(self, five_closes):
        moving_average = sma(five_closes, 3)

        assert math.isnan(moving_average[0])
        assert math.isnan(moving_average[1])
        assert moving_average[2] == pytest.approx(2.0)
        assert moving_average[3] == pytest.approx(3.0)
        assert moving_average[4] == pytest.approx(4.0)


class TestEma:
    def test_period_three_uses_alpha_half(self, three_closes):
        moving_average = ema(three_closes, 3)

        assert moving_average[0] == pytest.approx(1.0)
        assert moving_average[1] == pytest.approx(1.5)
        assert moving_average[2] == pytest.approx(2.25)


class TestWma:
    def test_period_three_applies_linear_weights(self, three_closes):
        moving_average = wma(three_closes, 3)

        assert math.isnan(moving_average[0])
        assert math.isnan(moving_average[1])
        assert moving_average[2] == pytest.approx((1 + 4 + 9) / 6)


class TestMovingAverage:
    @pytest.mark.parametrize(
        ("ma_type", "reference_average"),
        [
            (MAType.SMA, sma),
            (MAType.EMA, ema),
            (MAType.WMA, wma),
        ],
    )
    def test_dispatches_to_matching_average(
        self, five_closes, ma_type, reference_average
    ):
        dispatched = moving_average(five_closes, 3, ma_type)

        np.testing.assert_array_equal(dispatched, reference_average(five_closes, 3))

    def test_defaults_to_sma(self, five_closes):
        default_average = moving_average(five_closes, 3)

        np.testing.assert_array_equal(default_average, sma(five_closes, 3))

    @pytest.mark.parametrize("ma_type", list(MAType))
    def test_returns_one_value_per_close(self, five_closes, ma_type):
        averaged = moving_average(five_closes, 3, ma_type)

        assert isinstance(averaged, np.ndarray)
        assert len(averaged) == len(five_closes)
