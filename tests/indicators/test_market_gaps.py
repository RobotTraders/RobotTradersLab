import math

import numpy as np
import pytest

from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.indicators import MAType, donchian_midline, ema, sma, trix, wma

PERIOD = 4
TRIX_LENGTH = 3
LONGER_THAN_THE_SERIES = 29


@pytest.fixture
def gapped_closes() -> np.ndarray:
    return np.array(
        [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, np.nan,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, np.nan, np.nan,
            14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, np.nan, np.nan,
            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, np.nan, np.nan,
        ]
    )  # fmt: skip


@pytest.fixture
def gapped_highs(gapped_closes: np.ndarray) -> np.ndarray:
    return gapped_closes + 1.0


@pytest.fixture
def gapped_lows(gapped_closes: np.ndarray) -> np.ndarray:
    return gapped_closes - 1.0


@pytest.fixture
def published(gapped_closes: np.ndarray) -> np.ndarray:
    return ~np.isnan(gapped_closes)


class TestSma:
    def test_window_spans_the_published_candles(self, gapped_closes, published):
        averaged = sma(gapped_closes, PERIOD)

        assert math.isnan(averaged[2])
        assert averaged[3] == pytest.approx((0 + 1 + 2 + 3) / 4)
        assert math.isnan(averaged[7])
        assert averaged[9] == pytest.approx((4 + 5 + 6 + 7) / 4)
        assert np.isfinite(averaged[published][PERIOD - 1 :]).all()

    def test_period_longer_than_the_published_candles(self, gapped_closes):
        with pytest.raises(StrategyCriticalError) as failure:
            sma(gapped_closes, LONGER_THAN_THE_SERIES)

        assert "SMA(29)" in str(failure.value)
        assert "carries 28 across 36 stamps" in str(failure.value)
        assert "8 of them holding no candle" in str(failure.value)


class TestEma:
    def test_gaps_age_the_average_by_nothing(self, gapped_closes, published):
        averaged = ema(gapped_closes, PERIOD)

        assert math.isnan(averaged[7])
        np.testing.assert_allclose(
            averaged[published], ema(gapped_closes[published], PERIOD)
        )


class TestWma:
    def test_window_spans_the_published_candles(self, gapped_closes, published):
        averaged = wma(gapped_closes, PERIOD)

        assert math.isnan(averaged[2])
        assert averaged[3] == pytest.approx((0 * 1 + 1 * 2 + 2 * 3 + 3 * 4) / 10)
        assert math.isnan(averaged[7])
        assert averaged[9] == pytest.approx((4 * 1 + 5 * 2 + 6 * 3 + 7 * 4) / 10)
        assert np.isfinite(averaged[published][PERIOD - 1 :]).all()

    def test_period_longer_than_the_published_candles(self, gapped_closes):
        with pytest.raises(StrategyCriticalError) as failure:
            wma(gapped_closes, LONGER_THAN_THE_SERIES)

        assert "WMA(29)" in str(failure.value)
        assert "carries 28 across 36 stamps" in str(failure.value)


class TestDonchianMidline:
    def test_channel_spans_the_published_candles(
        self, gapped_highs, gapped_lows, published
    ):
        midline = donchian_midline(gapped_highs, gapped_lows, PERIOD)

        assert math.isnan(midline[2])
        assert midline[3] == pytest.approx((4.0 + -1.0) / 2)
        assert math.isnan(midline[7])
        assert midline[9] == pytest.approx((8.0 + 3.0) / 2)
        assert np.isfinite(midline[published][PERIOD - 1 :]).all()

    def test_a_candle_missing_one_of_its_extremes(self, gapped_highs, gapped_lows):
        gapped_highs[3] = np.nan
        gapped_lows[5] = np.nan

        midline = donchian_midline(gapped_highs, gapped_lows, PERIOD)

        assert math.isnan(midline[3])
        assert math.isnan(midline[5])
        assert midline[4] == pytest.approx((5.0 + -1.0) / 2)
        assert midline[6] == pytest.approx((7.0 + 0.0) / 2)

    def test_period_longer_than_the_published_candles(self, gapped_highs, gapped_lows):
        with pytest.raises(StrategyCriticalError) as failure:
            donchian_midline(gapped_highs, gapped_lows, LONGER_THAN_THE_SERIES)

        assert "DCM(29)" in str(failure.value)
        assert "carries 28 across 36 stamps" in str(failure.value)


class TestTrix:
    def test_the_candle_after_a_gap_is_read_from_the_one_before_it(self, gapped_closes):
        computed = trix(gapped_closes, TRIX_LENGTH, PERIOD)

        assert math.isnan(computed.trix[7])
        assert np.isfinite(computed.trix[9])
        assert np.isfinite(computed.histogram[9])

    def test_a_windowed_signal_average_needing_more_than_the_series_holds(
        self, gapped_closes
    ):
        with pytest.raises(StrategyCriticalError) as failure:
            trix(
                gapped_closes,
                TRIX_LENGTH,
                LONGER_THAN_THE_SERIES,
                signal_type=MAType.SMA,
            )

        assert "SMA(29)" in str(failure.value)
        assert "carries 27 across 36 stamps" in str(failure.value)
        assert "9 of them holding no candle" in str(failure.value)
