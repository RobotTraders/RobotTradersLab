from typing import Any

from robottraderslab.indicators import MAType, moving_average, trix
from robottraderslab.strategies import Candles, ChartLine

from .profile_config import (
    DEFAULT_SIGNAL_TYPE,
    DEFAULT_TREND_LENGTH,
    DEFAULT_TREND_TYPE,
)


def get_lightweight_chart_indicators(
    candles: Candles,
    indicator_params: dict[str, Any],
) -> list[ChartLine]:
    """Draw the trend average beside the candles and TRIX below them.

    Args:
        indicator_params: `trix_length` and `signal_length` as a profile
            requires them; `signal_type`, `trend_length` and `trend_type`
            fall back to the profile defaults.
    """
    close = candles.close
    trix_length = indicator_params["trix_length"]
    signal_length = indicator_params["signal_length"]
    signal_type = MAType(indicator_params.get("signal_type", DEFAULT_SIGNAL_TYPE))
    trend_length = indicator_params.get("trend_length", DEFAULT_TREND_LENGTH)
    trend_type = MAType(indicator_params.get("trend_type", DEFAULT_TREND_TYPE))

    indicator = trix(
        close,
        trix_length,
        signal_length,
        signal_type=signal_type,
    )
    return [
        ChartLine(
            name=f"Trend {trend_type} {trend_length}",
            values=moving_average(close, trend_length, trend_type),
            colour="orange",
        ),
        ChartLine(
            name="TRIX",
            values=indicator.trix,
            colour="blue",
            pane="separate",
        ),
        ChartLine(
            name="Signal",
            values=indicator.signal,
            colour="red",
            pane="separate",
        ),
        ChartLine(
            name="Histogram",
            values=indicator.histogram,
            colour="gray",
            pane="separate",
            shape="histogram",
        ),
    ]
