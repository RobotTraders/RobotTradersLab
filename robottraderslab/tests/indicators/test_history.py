import numpy as np
import pytest

from robottraderslab.indicators import value_periods_ago


def test_each_candle_reads_the_value_one_period_ago():
    series = np.array([10.0, 11.0, 13.0, 12.0])

    past = value_periods_ago(series, 1)

    assert np.isnan(past[0])
    assert past[1:].tolist() == [10.0, 11.0, 13.0]
    assert past.dtype == np.float64


def test_each_candle_reads_the_value_two_periods_ago():
    series = np.array([10.0, 11.0, 13.0, 12.0])

    past = value_periods_ago(series, 2)

    assert np.isnan(past[:2]).all()
    assert past[2:].tolist() == [10.0, 11.0]


def test_periods_beyond_the_series_read_nan_on_every_candle():
    series = np.array([10.0, 11.0, 13.0])

    past = value_periods_ago(series, 5)

    assert past.shape == (3,)
    assert np.isnan(past).all()


def test_an_empty_series_gives_an_empty_series():
    past = value_periods_ago(np.array([], dtype=np.float64), 1)

    assert past.shape == (0,)
    assert past.dtype == np.float64


def test_a_candle_reading_a_stamp_the_market_left_empty_reads_nan():
    series = np.array([1.0, np.nan, 3.0, 4.0])

    past = value_periods_ago(series, 1)

    assert np.isnan(past[2])
    assert past[3] == 3.0


def test_the_series_is_left_untouched():
    series = np.array([1.0, 2.0, 3.0])

    value_periods_ago(series, 1)

    assert series.tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("periods", [0, -1])
def test_a_period_count_below_one_is_refused(periods: int):
    with pytest.raises(ValueError, match="periods must be at least 1"):
        value_periods_ago(np.array([1.0, 2.0]), periods)
