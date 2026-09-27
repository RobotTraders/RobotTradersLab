import numpy as np
import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import MissingOhlcvDataError
from robottraderslab.strategies import OHLCVs

BTC = Symbol.create("BTC/USDT:USDT")
TIMEFRAMES = ("15m", "1h", "4h")
PANDAS_FREQUENCIES = {"15m": "15min", "1h": "1h", "4h": "4h"}


def _price_path(seed: int, first_open: str, periods: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.normal(0.0, 1.0, periods))
    opens = np.concatenate([[100.0], closes[:-1]])
    spread = np.abs(rng.normal(0.0, 0.5, periods))
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + spread,
            "low": np.minimum(opens, closes) - spread,
            "close": closes,
            "volume": rng.uniform(1.0, 10.0, periods),
        },
        index=pd.date_range(first_open, periods=periods, freq="15min"),
    )


def _aggregate(quarter_hours: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return (
        quarter_hours.resample(PANDAS_FREQUENCIES[timeframe])
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )


def _candles_from_one_path(seed: int) -> dict[str, pd.DataFrame]:
    quarter_hours = _price_path(seed, "2024-01-01 00:00", periods=4 * 24 * 3)
    return {
        "15m": quarter_hours.loc["2024-01-01 03:45":],
        "1h": _aggregate(quarter_hours, "1h").loc["2024-01-01 03:00":],
        "4h": _aggregate(quarter_hours, "4h"),
    }


def _close_seconds(frame: pd.DataFrame, timeframe: str) -> np.ndarray:
    closes = frame.index + pd.Timedelta(PANDAS_FREQUENCIES[timeframe])
    return closes.to_numpy().astype("datetime64[s]").astype(float)


def _last_closed_is_odd(closes: np.ndarray, moment: float) -> bool:
    last_closed = int((closes <= moment).sum()) - 1
    return last_closed % 2 == 1


@pytest.fixture
def hour_and_four_hours() -> OHLCVs:
    """1h and 4h candles of one symbol from 16:00 to 24:00, the 4h built by
    aggregating the 1h.
    """
    hourly = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "high": [11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5],
            "low": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5],
            "close": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            "volume": [1.0] * 8,
        },
        index=pd.date_range("2024-01-01 16:00", periods=8, freq="1h"),
    )
    four_hourly = _aggregate(hourly, "4h")
    return OHLCVs({"1h": {BTC: hourly}, "4h": {BTC: four_hourly}})


class TestEveryReadClosedByTheIteration:
    @pytest.mark.parametrize("seed", [1, 7, 42])
    def test_every_readable_candle_closed_by_the_iteration(self, seed):
        candles = _candles_from_one_path(seed)
        closes_by_tf = {tf: _close_seconds(candles[tf], tf) for tf in TIMEFRAMES}
        ohlcvs = OHLCVs({tf: {BTC: frame} for tf, frame in candles.items()})
        for timeframe, closes in closes_by_tf.items():
            ohlcvs.add_column(BTC, timeframe, "closed_at", closes)
            ohlcvs.add_column(BTC, timeframe, "odd", np.arange(closes.size) % 2 == 1)

        for snapshot in ohlcvs._iter_timeframes():
            moment = pd.Timestamp(snapshot.timestamp).timestamp()
            for timeframe, closes in closes_by_tf.items():
                assert ohlcvs.current(BTC, timeframe, "closed_at") <= moment
                assert ohlcvs.signal(BTC, timeframe, "odd") == _last_closed_is_odd(
                    closes, moment
                )
            if "4h" in snapshot.triggered_timeframes:
                assert ohlcvs.current(BTC, "4h", "close") == ohlcvs.current(
                    BTC, "1h", "close"
                )
                assert ohlcvs.current(BTC, "1h", "close") == ohlcvs.current(
                    BTC, "15m", "close"
                )


class TestTwoTimeframes:
    def test_the_higher_timeframe_reads_its_last_closed_candle(
        self, hour_and_four_hours
    ):
        four_hour_closes = []

        for _ in hour_and_four_hours._iter_timeframes(
            after=pd.Timestamp("2024-01-01 19:00")
        ):
            four_hour_closes.append(hour_and_four_hours.current(BTC, "4h", "close"))

        assert four_hour_closes == [14.0, 14.0, 14.0, 14.0, 18.0]

    def test_both_trigger_on_their_shared_close_with_equal_closes(
        self, hour_and_four_hours
    ):
        shared_close = next(
            snapshot
            for snapshot in hour_and_four_hours._iter_timeframes()
            if snapshot.timestamp == pd.Timestamp("2024-01-01 20:00")
        )

        assert shared_close.triggered_timeframes == ["1h", "4h"]
        assert hour_and_four_hours.current(BTC, "4h", "close") == 14.0
        assert hour_and_four_hours.current(BTC, "1h", "close") == 14.0

    def test_the_iterations_are_the_moments_candles_closed(self, hour_and_four_hours):
        moments = [
            snapshot.timestamp for snapshot in hour_and_four_hours._iter_timeframes()
        ]

        assert moments == list(pd.date_range("2024-01-01 17:00", periods=8, freq="1h"))

    def test_a_timeframe_before_its_first_close(self, hour_and_four_hours):
        next(hour_and_four_hours._iter_timeframes())

        with pytest.raises(
            MissingOhlcvDataError,
            match="BTC/USDT:USDT@4h: no candle had closed yet",
        ):
            hour_and_four_hours.current(BTC, "4h", "close")


class TestSimulatedCandle:
    def test_is_stamped_with_its_close(self, hour_and_four_hours):
        first = next(hour_and_four_hours._iter_timeframes())

        assert first.ohlcvs_by_symbol[BTC].timestamp == pd.Timestamp("2024-01-01 17:00")

    def test_a_symbol_on_both_timeframes_takes_the_shorter_candle(
        self, hour_and_four_hours
    ):
        shared_close = next(
            snapshot
            for snapshot in hour_and_four_hours._iter_timeframes()
            if snapshot.timestamp == pd.Timestamp("2024-01-01 20:00")
        )

        simulated = shared_close.ohlcvs_by_symbol[BTC]

        assert simulated.open == 13.0
        assert simulated.low == 12.5

    def test_a_longer_candle_past_the_shorter_series_steps_nothing(self):
        hourly = pd.DataFrame(
            {column: [10.0] * 5 for column in ("open", "high", "low", "close")}
            | {"volume": [1.0] * 5},
            index=pd.date_range("2024-01-01 16:00", periods=5, freq="1h"),
        )
        four_hourly = pd.DataFrame(
            {column: [10.0, 12.0] for column in ("open", "high", "low", "close")}
            | {"volume": [1.0, 1.0]},
            index=pd.date_range("2024-01-01 16:00", periods=2, freq="4h"),
        )
        ohlcvs = OHLCVs({"1h": {BTC: hourly}, "4h": {BTC: four_hourly}})

        last = list(ohlcvs._iter_timeframes())[-1]

        assert last.triggered_timeframes == ["4h"]
        assert BTC not in last.ohlcvs_by_symbol
