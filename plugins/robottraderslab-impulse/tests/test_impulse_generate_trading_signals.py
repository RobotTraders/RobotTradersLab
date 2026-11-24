import numpy as np
import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import MissingOhlcvDataError
from robottraderslab.indicators import MAType
from robottraderslab.strategies import OHLCVs, TradingMode
from robottraderslab_impulse.impulse_indicator import compute_signals

BTC = Symbol.create("BTC/USDT:USDT")
TIMEFRAME = "1d"
SETUP_ID = "BTC/USDT:USDT@1d-alpha"
SIGNAL_SUFFIXES = ("long_entry", "long_exit", "short_entry", "short_exit")

RISING_CLOSES = [100.0 + 2.0 * candle for candle in range(40)]


def _build_ohlcvs(close: list[float]) -> OHLCVs:
    dates = pd.date_range("2024-01-01", periods=len(close), freq="1D")
    frame = pd.DataFrame(
        {
            "open": close,
            "high": [price + 1 for price in close],
            "low": [price - 1 for price in close],
            "close": close,
            "volume": [1000.0] * len(close),
        },
        index=dates,
    )
    return OHLCVs({TIMEFRAME: {BTC: frame}})


def _profile(**overrides: object) -> dict:
    profile: dict = {
        "symbol": "BTC/USDT:USDT",
        "timeframe": TIMEFRAME,
        "total_balance_ratio": 1.0,
        "trix_length": 5,
        "signal_length": 3,
        "trend_length": 10,
        "tag": "alpha",
    }
    profile.update(overrides)
    return profile


def _collect_values(ohlcvs: OHLCVs, column: str) -> list[float]:
    values: list[float] = []
    for _ in ohlcvs._iter_timeframes():
        values.append(ohlcvs.current(BTC, TIMEFRAME, column))
    return values


def _ohlcvs_without_the_profile_pair() -> OHLCVs:
    eth = Symbol.create("ETH/USDT:USDT")
    dates = pd.date_range("2024-01-01", periods=5, freq="1D")
    frame = pd.DataFrame(
        {
            "open": [1.0] * 5,
            "high": [2.0] * 5,
            "low": [0.5] * 5,
            "close": [1.0] * 5,
            "volume": [1000.0] * 5,
        },
        index=dates,
    )
    return OHLCVs({TIMEFRAME: {eth: frame}})


class TestGenerateTradingSignals:
    def test_a_missing_pair_propagates_in_backtest(self, make_strategy):
        strategy = make_strategy([_profile()])
        ohlcvs = _ohlcvs_without_the_profile_pair()

        with pytest.raises(MissingOhlcvDataError):
            strategy.generate_trading_signals(ohlcvs)

    def test_a_missing_pair_skips_the_profile_in_live(self, make_strategy, caplog):
        strategy = make_strategy([_profile()], trading_mode=TradingMode.LIVE)
        ohlcvs = _ohlcvs_without_the_profile_pair()

        strategy.generate_trading_signals(ohlcvs)

        assert "sat out this candle" in caplog.text

    def test_adds_a_column_per_side_and_direction(self, make_strategy):
        strategy = make_strategy([_profile()])
        ohlcvs = _build_ohlcvs(RISING_CLOSES)

        strategy.generate_trading_signals(ohlcvs)

        for suffix in SIGNAL_SUFFIXES:
            values = _collect_values(ohlcvs, f"{SETUP_ID}_{suffix}")
            assert len(values) == len(RISING_CLOSES)

    def test_columns_carry_the_computed_signals(self, make_strategy):
        strategy = make_strategy([_profile()])
        ohlcvs = _build_ohlcvs(RISING_CLOSES)
        signals = compute_signals(
            np.array(RISING_CLOSES), 5, 3, MAType.SMA, 10, MAType.SMA
        )

        strategy.generate_trading_signals(ohlcvs)

        long_entries = _collect_values(ohlcvs, f"{SETUP_ID}_long_entry")
        assert any(long_entries)
        assert long_entries == [float(flag) for flag in signals.long_entry]

    def test_the_signal_type_reaches_the_computation(self, make_strategy):
        strategy = make_strategy([_profile(signal_type="WMA")])
        ohlcvs = _build_ohlcvs(RISING_CLOSES)
        signals = compute_signals(
            np.array(RISING_CLOSES), 5, 3, MAType.WMA, 10, MAType.SMA
        )

        strategy.generate_trading_signals(ohlcvs)

        long_entries = _collect_values(ohlcvs, f"{SETUP_ID}_long_entry")
        assert long_entries == [float(flag) for flag in signals.long_entry]

    def test_stacked_profiles_get_their_own_columns(self, make_strategy):
        strategy = make_strategy(
            [
                _profile(tag="fast"),
                _profile(tag="slow", trix_length=8),
            ]
        )
        ohlcvs = _build_ohlcvs(RISING_CLOSES)

        strategy.generate_trading_signals(ohlcvs)

        for tag in ("fast", "slow"):
            values = _collect_values(ohlcvs, f"BTC/USDT:USDT@1d-{tag}_long_entry")
            assert len(values) == len(RISING_CLOSES)
