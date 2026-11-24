import numpy as np
import pandas as pd
import pytest

from robottraderslab.indicators import MAType, ema, sma, trix

LENGTH = 5
SIGNAL_LENGTH = 3


@pytest.fixture
def close() -> np.ndarray:
    return np.array([100.0 + i * 1.5 for i in range(30)])


class TestTripleEma:
    def test_matches_nested_ema(self, close):
        computed = trix(close, LENGTH, SIGNAL_LENGTH)

        expected = ema(ema(ema(close, LENGTH), LENGTH), LENGTH)
        np.testing.assert_array_equal(computed.triple_ema, expected)


class TestTrix:
    def test_is_pct_change_of_triple_ema_times_hundred(self, close):
        computed = trix(close, LENGTH, SIGNAL_LENGTH)

        expected = pd.Series(computed.triple_ema).pct_change().to_numpy() * 100
        np.testing.assert_allclose(computed.trix, expected)


class TestSignal:
    def test_default_signal_uses_ema(self, close):
        computed = trix(close, LENGTH, SIGNAL_LENGTH)

        expected = ema(computed.trix, SIGNAL_LENGTH)
        np.testing.assert_array_equal(computed.signal, expected)

    def test_custom_signal_average_is_applied(self, close):
        computed = trix(close, LENGTH, SIGNAL_LENGTH, signal_type=MAType.SMA)

        expected = sma(computed.trix, SIGNAL_LENGTH)
        np.testing.assert_array_equal(computed.signal, expected)

    def test_sma_signal_differs_from_ema_signal(self, close):
        ema_result = trix(close, LENGTH, SIGNAL_LENGTH)
        sma_result = trix(close, LENGTH, SIGNAL_LENGTH, signal_type=MAType.SMA)

        assert not np.array_equal(ema_result.signal, sma_result.signal, equal_nan=True)
        assert not np.array_equal(
            ema_result.histogram, sma_result.histogram, equal_nan=True
        )


class TestHistogram:
    def test_is_trix_minus_signal(self, close):
        computed = trix(close, LENGTH, SIGNAL_LENGTH)

        expected = computed.trix - computed.signal
        np.testing.assert_array_equal(computed.histogram, expected)
