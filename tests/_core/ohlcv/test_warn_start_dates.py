import logging
from datetime import datetime, timezone

import pandas as pd

from robottraderslab import Symbol
from robottraderslab.strategies import OHLCVs

BTC = Symbol.create("BTC/USDT:USDT")
SHIB = Symbol.create("SHIB/USDT:USDT")


def _make_ohlcv(
    index: pd.DatetimeIndex,
    close_values: list[float | None],
) -> pd.DataFrame:
    close = pd.array(close_values, dtype="Float64")
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": close,
        },
        index=index,
    )


START_AFTER = datetime(2024, 1, 1, tzinfo=timezone.utc)
DATES_UTC = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")


class TestDataStartsLate:
    def test_no_warning_when_data_starts_at_configured_date(self, caplog):
        symbol_dict = {BTC: _make_ohlcv(DATES_UTC, [100.0] * 10)}

        with caplog.at_level(logging.WARNING):
            OHLCVs({"1d": symbol_dict})._warn_if_data_starts_late(START_AFTER)

        assert caplog.records == []

    def test_warns_when_data_starts_after_configured_date(self, caplog):
        closes: list[float | None] = [None] * 5 + [100.0] * 5
        symbol_dict = {BTC: _make_ohlcv(DATES_UTC, closes)}

        with caplog.at_level(logging.WARNING):
            OHLCVs({"1d": symbol_dict})._warn_if_data_starts_late(START_AFTER)

        assert len(caplog.records) == 1
        assert "no data on exchange before 2024-01-06" in caplog.records[0].message
        assert "backtest starts from there" in caplog.records[0].message

    def test_warns_when_no_data_at_all(self, caplog):
        symbol_dict = {BTC: _make_ohlcv(DATES_UTC, [None] * 10)}

        with caplog.at_level(logging.WARNING):
            OHLCVs({"1d": symbol_dict})._warn_if_data_starts_late(START_AFTER)

        assert len(caplog.records) == 1
        assert "no data on exchange" in caplog.records[0].message
        assert "skipped in backtest" in caplog.records[0].message

    def test_warns_per_symbol(self, caplog):
        btc_closes: list[float | None] = [None] * 3 + [100.0] * 7
        shib_closes: list[float | None] = [None] * 7 + [0.01] * 3
        symbol_dict = {
            BTC: _make_ohlcv(DATES_UTC, btc_closes),
            SHIB: _make_ohlcv(DATES_UTC, shib_closes),
        }

        with caplog.at_level(logging.WARNING):
            OHLCVs({"1d": symbol_dict})._warn_if_data_starts_late(START_AFTER)

        assert len(caplog.records) == 2
        assert "BTC/USDT:USDT@1d" in caplog.records[0].message
        assert "2024-01-04" in caplog.records[0].message
        assert "SHIB/USDT:USDT@1d" in caplog.records[1].message
        assert "2024-01-08" in caplog.records[1].message

    def test_ignores_lookback_data_before_start(self, caplog):
        lookback_dates = pd.date_range("2023-12-25", periods=17, freq="D", tz="UTC")
        closes: list[float | None] = [100.0] * 7 + [None] * 5 + [100.0] * 5
        symbol_dict = {BTC: _make_ohlcv(lookback_dates, closes)}

        with caplog.at_level(logging.WARNING):
            OHLCVs({"1d": symbol_dict})._warn_if_data_starts_late(START_AFTER)

        assert len(caplog.records) == 1
        assert "2024-01-06" in caplog.records[0].message
