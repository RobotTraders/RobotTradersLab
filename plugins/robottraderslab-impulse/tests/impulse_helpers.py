from collections.abc import Mapping

from robottraderslab import Symbol, TimeFrame
from robottraderslab.strategies import OHLCVs

SIGNAL_NAMES = ("long_entry", "long_exit", "short_entry", "short_exit")


def write_signal_columns(
    ohlcvs: OHLCVs,
    symbol: Symbol,
    timeframe: TimeFrame,
    profile_id: str,
    scripted: Mapping[int, str],
) -> None:
    """Write the four signal columns of a profile over every candle of its series.

    `scripted` maps a candle's index to the signal it raises, so a column holds
    1.0 on the candles scripted with its name and 0.0 on every other.
    """
    candles = len(ohlcvs.column(symbol, timeframe, "close"))
    for name in SIGNAL_NAMES:
        ohlcvs.add_column(
            symbol,
            timeframe,
            f"{profile_id}_{name}",
            [float(scripted.get(candle) == name) for candle in range(candles)],
        )
