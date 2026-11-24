import numpy as np

from robottraderslab.indicators import crossover, crossunder

FAST = np.array([np.nan, 1.0, 3.0, 4.0, 2.0, 1.0, 2.0])
SLOW = np.array([np.nan, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])


def test_crossover_is_true_only_on_the_candle_the_series_moves_above():
    assert crossover(FAST, SLOW).tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]


def test_crossunder_is_true_only_on_the_candle_the_series_moves_below():
    assert crossunder(FAST, SLOW).tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
    ]


def test_touching_the_other_series_is_no_crossing():
    series = np.array([1.0, 2.0, 2.0, 1.0])
    other = np.full(4, 2.0)

    assert not crossover(series, other).any()
    assert crossunder(series, other).tolist() == [False, False, False, True]


def test_a_candle_after_a_missing_value_never_crosses():
    series = np.array([1.0, np.nan, 3.0])
    other = np.full(3, 2.0)

    assert not crossover(series, other).any()
