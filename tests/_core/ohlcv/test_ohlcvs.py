import math
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import (
    MissingOhlcvDataError,
    StrategyCriticalError,
)
from robottraderslab.strategies import OHLCVs


def _make_ohlcv_dict(
    timestamps: list[str],
    symbols: list[Symbol],
    close_values: list[list[float]],
) -> dict[Symbol, pd.DataFrame]:
    """Build a per-symbol dict of DataFrames matching the assembler format.

    Args:
        timestamps: ISO timestamp strings for the index.
        symbols: Symbols to include as keys.
        close_values: One inner list per symbol, each with len(timestamps) values.
    """
    index = pd.DatetimeIndex([pd.Timestamp(t) for t in timestamps])
    return {
        symbol: pd.DataFrame(
            {
                "open": symbol_close,
                "high": symbol_close,
                "low": symbol_close,
                "close": symbol_close,
                "volume": [1.0] * len(timestamps),
            },
            index=index,
        )
        for symbol, symbol_close in zip(symbols, close_values)
    }


@pytest.fixture
def btc() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def eth() -> Symbol:
    return Symbol.create("ETH/USDT:USDT")


@pytest.fixture
def ohlcvs_on_first_candle(btc: Symbol) -> OHLCVs:
    """One BTC daily candle, with the strategy standing on it."""
    ohlcvs = OHLCVs({"1d": _make_ohlcv_dict(["2024-01-01"], [btc], [[100.0]])})
    next(ohlcvs._iter_timeframes())
    return ohlcvs


class TestOhlcvsBySymbol:
    def test_single_timeframe(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [btc],
            [[100.0, 200.0, 300.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        snapshots = list(ohlcvs._iter_timeframes())

        assert len(snapshots) == 3
        assert snapshots[0].ohlcvs_by_symbol[btc].close == 100.0
        assert snapshots[1].ohlcvs_by_symbol[btc].close == 200.0
        assert snapshots[2].ohlcvs_by_symbol[btc].close == 300.0

    def test_multi_timeframe(self, btc):
        hourly_ts = [
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 02:00",
            "2024-01-01 03:00",
        ]
        four_hourly_ts = ["2024-01-01 00:00"]
        symbol_dict_1h = _make_ohlcv_dict(hourly_ts, [btc], [[100.0, 20.0, 30.0, 40.0]])
        symbol_dict_4h = _make_ohlcv_dict(four_hourly_ts, [btc], [[100.0]])
        ohlcvs = OHLCVs({"1h": symbol_dict_1h, "4h": symbol_dict_4h})

        snapshots = list(ohlcvs._iter_timeframes())

        assert len(snapshots) == 4
        assert snapshots[0].ohlcvs_by_symbol[btc].close == 100.0
        assert snapshots[1].ohlcvs_by_symbol[btc].close == 20.0
        assert snapshots[2].ohlcvs_by_symbol[btc].close == 30.0
        assert snapshots[3].ohlcvs_by_symbol[btc].close == 40.0

    def test_multiple_symbols(self, btc, eth):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02"],
            [btc, eth],
            [[100.0, 200.0], [10.0, 20.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        snapshots = list(ohlcvs._iter_timeframes())

        assert snapshots[0].ohlcvs_by_symbol[btc].close == 100.0
        assert snapshots[0].ohlcvs_by_symbol[eth].close == 10.0
        assert snapshots[1].ohlcvs_by_symbol[btc].close == 200.0
        assert snapshots[1].ohlcvs_by_symbol[eth].close == 20.0

    def test_a_start_on_a_candle_open_decides_first_on_its_close(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [btc],
            [[100.0, 200.0, 300.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        snapshots = list(
            ohlcvs._iter_timeframes(after=datetime.fromisoformat("2024-01-02"))
        )

        assert len(snapshots) == 2
        assert snapshots[0].ohlcvs_by_symbol[btc].close == 200.0
        assert snapshots[1].ohlcvs_by_symbol[btc].close == 300.0

    def test_start_idx_negative_one(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [btc],
            [[100.0, 200.0, 300.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        snapshots = list(ohlcvs._iter_timeframes(start_idx=-1))

        assert len(snapshots) == 1
        assert snapshots[0].ohlcvs_by_symbol[btc].close == 300.0


class TestCurrent:
    def test_returns_correct_scalar(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [btc],
            [[100.0, 200.0, 300.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})
        close_values = []

        for _ in ohlcvs._iter_timeframes():
            close_values.append(ohlcvs.current(btc, "1d", "close"))

        assert close_values == [100.0, 200.0, 300.0]

    def test_with_multiple_symbols(self, btc, eth):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02"],
            [btc, eth],
            [[100.0, 200.0], [10.0, 20.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})
        btc_closes = []
        eth_closes = []

        for _ in ohlcvs._iter_timeframes():
            btc_closes.append(ohlcvs.current(btc, "1d", "close"))
            eth_closes.append(ohlcvs.current(eth, "1d", "close"))

        assert btc_closes == [100.0, 200.0]
        assert eth_closes == [10.0, 20.0]

    def test_with_nan(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02"],
            [btc],
            [[float("nan"), 200.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})
        close_values = []

        for _ in ohlcvs._iter_timeframes():
            close_values.append(ohlcvs.current(btc, "1d", "close"))

        assert math.isnan(close_values[0])
        assert close_values[1] == 200.0

    def test_with_strategy_added_dataframe_column(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [btc],
            [[100.0, 200.0, 300.0]],
        )
        symbol_dict[btc]["buy"] = [False, True, False]
        ohlcvs = OHLCVs({"1d": symbol_dict})
        buy_signals = []

        for _ in ohlcvs._iter_timeframes():
            buy_signals.append(ohlcvs.current(btc, "1d", "buy"))

        assert buy_signals == [0.0, 1.0, 0.0]

    def test_a_name_no_column_carries(self, btc, ohlcvs_on_first_candle):
        with pytest.raises(
            KeyError,
            match="BTC/USDT:USDT@1d has no column 'nope'; "
            "its columns are open, high, low, close, volume",
        ):
            ohlcvs_on_first_candle.current(btc, "1d", "nope")


class TestSignal:
    def test_reads_a_boolean_column(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [btc],
            [[100.0, 200.0, 300.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})
        ohlcvs.add_column(btc, "1d", "buy", [False, True, False])
        raised = []

        for _ in ohlcvs._iter_timeframes():
            raised.append(ohlcvs.signal(btc, "1d", "buy"))

        assert raised == [False, True, False]

    def test_reads_a_column_written_as_numbers(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02"], [btc], [[100.0, 200.0]]
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})
        ohlcvs.add_column(btc, "1d", "buy", [0.0, 1.0])
        raised = []

        for _ in ohlcvs._iter_timeframes():
            raised.append(ohlcvs.signal(btc, "1d", "buy"))

        assert raised == [False, True]

    def test_a_nan_is_not_raised(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02"], [btc], [[100.0, 200.0]]
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})
        ohlcvs.add_column(btc, "1d", "buy", [float("nan"), 2.5])
        raised = []

        for _ in ohlcvs._iter_timeframes():
            raised.append(ohlcvs.signal(btc, "1d", "buy"))

        assert raised == [False, True]


class TestColumn:
    def test_returns_numpy_array(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [btc],
            [[100.0, 200.0, 300.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        close_column = ohlcvs.column(btc, "1d", "close")

        assert isinstance(close_column, np.ndarray)
        assert list(close_column) == [100.0, 200.0, 300.0]

    def test_returns_correct_column(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02"],
            [btc],
            [[100.0, 200.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        volume_column = ohlcvs.column(btc, "1d", "volume")

        assert list(volume_column) == [1.0, 1.0]

    def test_with_multiple_symbols(self, btc, eth):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02"],
            [btc, eth],
            [[100.0, 200.0], [10.0, 20.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        btc_close = ohlcvs.column(btc, "1d", "close")
        eth_close = ohlcvs.column(eth, "1d", "close")

        assert list(btc_close) == [100.0, 200.0]
        assert list(eth_close) == [10.0, 20.0]

    def test_with_multiple_timeframes(self, btc):
        hourly = _make_ohlcv_dict(
            ["2024-01-01 00:00", "2024-01-01 01:00"],
            [btc],
            [[100.0, 200.0]],
        )
        daily = _make_ohlcv_dict(
            ["2024-01-01"],
            [btc],
            [[500.0]],
        )
        ohlcvs = OHLCVs({"1h": hourly, "1d": daily})

        hourly_close = ohlcvs.column(btc, "1h", "close")
        daily_close = ohlcvs.column(btc, "1d", "close")

        assert list(hourly_close) == [100.0, 200.0]
        assert list(daily_close) == [500.0]

    def test_a_name_no_column_carries(self, btc):
        symbol_dict = _make_ohlcv_dict(["2024-01-01"], [btc], [[100.0]])
        ohlcvs = OHLCVs({"1d": symbol_dict})

        with pytest.raises(
            KeyError,
            match="BTC/USDT:USDT@1d has no column 'nope'; "
            "its columns are open, high, low, close, volume",
        ):
            ohlcvs.column(btc, "1d", "nope")


class TestCovers:
    def test_a_moment_inside_the_window(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02"], [btc], [[100.0, 200.0]]
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        assert ohlcvs.covers(btc, "1d", datetime(2024, 1, 2, tzinfo=UTC))

    def test_the_close_of_the_last_candle(self, btc):
        symbol_dict = _make_ohlcv_dict(["2024-01-01"], [btc], [[100.0]])
        ohlcvs = OHLCVs({"1d": symbol_dict})

        assert ohlcvs.covers(btc, "1d", datetime(2024, 1, 2, tzinfo=UTC))

    def test_a_moment_after_the_last_candle_closed(self, btc):
        symbol_dict = _make_ohlcv_dict(["2024-01-01"], [btc], [[100.0]])
        ohlcvs = OHLCVs({"1d": symbol_dict})

        assert not ohlcvs.covers(btc, "1d", datetime(2024, 1, 2, 0, 1, tzinfo=UTC))

    def test_a_moment_before_the_first_candle(self, btc):
        symbol_dict = _make_ohlcv_dict(["2024-01-02"], [btc], [[200.0]])
        ohlcvs = OHLCVs({"1d": symbol_dict})

        assert not ohlcvs.covers(btc, "1d", datetime(2024, 1, 1, tzinfo=UTC))

    def test_an_aware_moment_is_read_in_utc(self, btc):
        symbol_dict = _make_ohlcv_dict(["2024-01-01 12:00"], [btc], [[100.0]])
        ohlcvs = OHLCVs({"1d": symbol_dict})
        just_before = datetime(2024, 1, 1, 13, 0, tzinfo=timezone(timedelta(hours=2)))

        assert not ohlcvs.covers(btc, "1d", just_before)


class TestCandlesBetween:
    def test_the_candle_holding_the_lower_bound_is_left_out(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [btc],
            [[100.0, 200.0, 300.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        candles = ohlcvs.candles_between(
            btc,
            "1d",
            datetime(2024, 1, 4, tzinfo=UTC),
            after=datetime(2024, 1, 1, 12, tzinfo=UTC),
        )

        assert list(ohlcvs.column(btc, "1d", "close")[candles]) == [200.0, 300.0]

    def test_the_candle_opening_on_the_lower_bound_is_included(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [btc],
            [[100.0, 200.0, 300.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        candles = ohlcvs.candles_between(
            btc,
            "1d",
            datetime(2024, 1, 4, tzinfo=UTC),
            after=datetime(2024, 1, 2, tzinfo=UTC),
        )

        assert list(ohlcvs.column(btc, "1d", "close")[candles]) == [200.0, 300.0]

    def test_the_candle_closing_on_the_upper_bound_is_included(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02"], [btc], [[100.0, 200.0]]
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        candles = ohlcvs.candles_between(btc, "1d", datetime(2024, 1, 2, tzinfo=UTC))

        assert list(ohlcvs.column(btc, "1d", "close")[candles]) == [100.0]

    def test_no_candles_when_the_stretch_holds_none(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02"], [btc], [[100.0, 200.0]]
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        candles = ohlcvs.candles_between(
            btc,
            "1d",
            datetime(2024, 1, 2, tzinfo=UTC),
            after=datetime(2024, 1, 2, tzinfo=UTC),
        )

        assert list(ohlcvs.column(btc, "1d", "close")[candles]) == []


class TestTimestamps:
    def test_each_candle_is_stamped_with_its_close(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01 00:00", "2024-01-01 04:00"], [btc], [[100.0, 200.0]]
        )
        ohlcvs = OHLCVs({"4h": symbol_dict})

        closes = ohlcvs.timestamps(btc, "4h")

        assert list(closes) == [
            np.datetime64("2024-01-01T04:00"),
            np.datetime64("2024-01-01T08:00"),
        ]

    def test_a_month_closes_on_the_first_of_the_next(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-02-01"], [btc], [[100.0, 200.0]]
        )
        ohlcvs = OHLCVs({"1M": symbol_dict})

        closes = ohlcvs.timestamps(btc, "1M")

        assert list(closes) == [
            np.datetime64("2024-02-01"),
            np.datetime64("2024-03-01"),
        ]

    def test_an_aware_index_is_read_in_utc(self, btc):
        symbol_dict = _make_ohlcv_dict(["2024-01-01 02:00+02:00"], [btc], [[100.0]])
        ohlcvs = OHLCVs({"1h": symbol_dict})

        closes = ohlcvs.timestamps(btc, "1h")

        assert list(closes) == [np.datetime64("2024-01-01T01:00")]

    def test_a_pair_the_run_did_not_load(self, eth, ohlcvs_on_first_candle):
        with pytest.raises(MissingOhlcvDataError, match="ETH/USDT:USDT@1d"):
            ohlcvs_on_first_candle.timestamps(eth, "1d")


class TestAddColumn:
    @pytest.mark.parametrize(
        "values",
        [
            np.array([1.0, 0.0]),
            [1.0, 0.0],
            pd.Series([1.0, 0.0]),
        ],
        ids=["numpy_array", "plain_list", "pandas_series"],
    )
    def test_with_array_like_input(self, btc, values):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02"],
            [btc],
            [[100.0, 200.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})
        ohlcvs.add_column(btc, "1d", "signal", values)
        signal_values = []

        for _ in ohlcvs._iter_timeframes():
            signal_values.append(ohlcvs.current(btc, "1d", "signal"))

        assert signal_values == [1.0, 0.0]

    def test_does_not_modify_dataframe(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02"],
            [btc],
            [[100.0, 200.0]],
        )
        original_columns = list(symbol_dict[btc].columns)
        ohlcvs = OHLCVs({"1d": symbol_dict})

        ohlcvs.add_column(btc, "1d", "signal", np.array([1.0, 0.0]))
        list(ohlcvs._iter_timeframes())

        assert list(symbol_dict[btc].columns) == original_columns

    @pytest.mark.parametrize(
        "values",
        [np.array([True, False]), np.array([1.0, 0.0]), np.array([1, 0])],
        ids=["boolean", "float", "integer"],
    )
    def test_column_comes_back_in_the_dtype_it_was_written(self, btc, values):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02"],
            [btc],
            [[100.0, 200.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        ohlcvs.add_column(btc, "1d", "signal", values)

        stored = ohlcvs.column(btc, "1d", "signal")
        assert stored.dtype == values.dtype
        assert list(stored) == list(values)

    def test_fewer_values_than_candles(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [btc],
            [[100.0, 200.0, 300.0]],
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        with pytest.raises(
            ValueError,
            match="BTC/USDT:USDT@1d: column 'signal' was given 2 values for 3 candles",
        ):
            ohlcvs.add_column(btc, "1d", "signal", np.array([1.0, 0.0]))

    def test_a_two_dimensional_column(self, btc):
        symbol_dict = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-02"], [btc], [[100.0, 200.0]]
        )
        ohlcvs = OHLCVs({"1d": symbol_dict})

        with pytest.raises(
            ValueError,
            match=r"column 'signal' was given values of shape \(2, 1\) for 2 candles",
        ):
            ohlcvs.add_column(btc, "1d", "signal", np.array([[1.0], [0.0]]))


class TestEarliestCandleOpen:
    def test_returns_earliest_across_timeframes(self, btc, eth):
        symbol_dict_1h = _make_ohlcv_dict(
            ["2024-01-02", "2024-01-03"], [btc], [[1.0, 2.0]]
        )
        symbol_dict_4h = _make_ohlcv_dict(
            ["2024-01-01", "2024-01-03"], [eth], [[1.0, 2.0]]
        )
        ohlcvs = OHLCVs({"1h": symbol_dict_1h, "4h": symbol_dict_4h})

        earliest = ohlcvs._earliest_candle_open()

        assert earliest == pd.Timestamp("2024-01-01")

    def test_raises_when_all_series_empty(self, btc):
        symbol_dict = _make_ohlcv_dict([], [btc], [[]])
        ohlcvs = OHLCVs({"1d": symbol_dict})

        with pytest.raises(StrategyCriticalError, match="No candles"):
            ohlcvs._earliest_candle_open()


class TestMissingPair:
    def test_reading_a_missing_pair_raises(self, btc, eth):
        data = _make_ohlcv_dict(["2024-01-01"], [btc], [[100.0]])
        ohlcvs = OHLCVs({"1d": data})

        with pytest.raises(MissingOhlcvDataError, match="ETH/USDT:USDT@1d"):
            ohlcvs.column(eth, "1d", "close")

    def test_reading_a_missing_pair_on_the_current_candle(
        self, eth, ohlcvs_on_first_candle
    ):
        with pytest.raises(MissingOhlcvDataError, match="ETH/USDT:USDT@1d"):
            ohlcvs_on_first_candle.current(eth, "1d", "close")
