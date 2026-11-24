import numpy as np

from robottraderslab.strategies import Candles
from robottraderslab_impulse.chart_indicators import get_lightweight_chart_indicators

CLOSES = np.array([100.0 + i * 2.0 for i in range(250)])
CANDLES = Candles(
    open=CLOSES,
    high=CLOSES * 1.01,
    low=CLOSES * 0.99,
    close=CLOSES,
    volume=np.ones_like(CLOSES),
)


def _named(lines):
    return {line.name: line for line in lines}


class TestGetLightweightChartIndicators:
    def test_names_every_line_it_draws(self):
        lines = _named(
            get_lightweight_chart_indicators(
                CANDLES,
                {"trix_length": 8, "signal_length": 15, "trend_length": 200},
            )
        )

        assert set(lines) == {"Trend SMA 200", "TRIX", "Signal", "Histogram"}

    def test_the_histogram_is_drawn_as_bars_not_a_curve(self):
        lines = _named(
            get_lightweight_chart_indicators(
                CANDLES, {"trix_length": 8, "signal_length": 15}
            )
        )

        assert lines["Histogram"].shape == "histogram"

    def test_the_trix_and_its_signal_stay_curves(self):
        lines = _named(
            get_lightweight_chart_indicators(
                CANDLES, {"trix_length": 8, "signal_length": 15}
            )
        )

        assert lines["TRIX"].shape == "line"
        assert lines["Signal"].shape == "line"

    def test_omitted_optional_keys_fall_back_to_the_profile_defaults(self):
        lines = _named(
            get_lightweight_chart_indicators(
                CANDLES, {"trix_length": 8, "signal_length": 15}
            )
        )

        assert "Trend SMA 200" in lines
        assert "TRIX" in lines

    def test_names_the_trend_line_after_its_average(self):
        lines = _named(
            get_lightweight_chart_indicators(
                CANDLES,
                {
                    "trix_length": 8,
                    "signal_length": 15,
                    "trend_type": "EMA",
                    "trend_length": 50,
                },
            )
        )

        assert "Trend EMA 50" in lines

    def test_the_signal_average_changes_the_signal_line(self):
        params = {"trix_length": 5, "signal_length": 3}

        sma_signal = _named(
            get_lightweight_chart_indicators(CANDLES, {**params, "signal_type": "SMA"})
        )["Signal"].values
        wma_signal = _named(
            get_lightweight_chart_indicators(CANDLES, {**params, "signal_type": "WMA"})
        )["Signal"].values

        assert not np.allclose(sma_signal, wma_signal, equal_nan=True)

    def test_the_signal_average_defaults_to_sma(self):
        params = {"trix_length": 5, "signal_length": 3}

        default = _named(get_lightweight_chart_indicators(CANDLES, params))[
            "Signal"
        ].values
        explicit = _named(
            get_lightweight_chart_indicators(CANDLES, {**params, "signal_type": "SMA"})
        )["Signal"].values

        assert np.allclose(default, explicit, equal_nan=True)

    def test_a_line_carries_one_value_per_candle(self):
        lines = get_lightweight_chart_indicators(
            CANDLES, {"trix_length": 8, "signal_length": 15, "trend_length": 10}
        )

        assert all(len(line.values) == len(CLOSES) for line in lines)

    def test_trix_draws_below_the_candles(self):
        lines = _named(
            get_lightweight_chart_indicators(
                CANDLES, {"trix_length": 8, "signal_length": 15}
            )
        )

        assert lines["TRIX"].pane == "separate"
        assert lines["Trend SMA 200"].pane == "price"
