from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.futures import FuturesAccount
from robottraderslab.ma_strategy.futures_ma_strategy import FuturesMAStrategy
from robottraderslab.strategies import OHLCVs, TradingSystem

BTC = Symbol.create("BTC/USDT:USDT")
FAST_LENGTH = 2
SLOW_LENGTH = 3
TREND_LENGTH = 6
BALANCE_RATIO = 0.25
CLOSES = [
    10.0,
    10.0,
    10.0,
    12.0,
    14.0,
    12.0,
    8.0,
    6.0,
    8.0,
    12.0,
    14.0,
    12.0,
    12.0,
    10.0,
]
CROSSING_UP_CANDLES = [3, 9]
CROSSING_DOWN_CANDLES = [6, 12]
LONG_ENTRY_CANDLES = [9]
SHORT_ENTRY_CANDLES = [6]


@pytest.fixture
def make_strategy(
    mock_account: FuturesAccount, backtest_trading_system: TradingSystem
) -> Callable[..., FuturesMAStrategy]:
    def make(**profile_fields: Any) -> FuturesMAStrategy:
        profile = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1d",
            "fast_ma_length": FAST_LENGTH,
            "slow_ma_length": SLOW_LENGTH,
            "available_balance_ratio": BALANCE_RATIO,
            **profile_fields,
        }
        return FuturesMAStrategy(
            account=mock_account,
            trading_system=backtest_trading_system,
            config_dir=Path("bot-config-dir"),
            profiles=[profile],
        )

    return make


@pytest.fixture
def ohlcvs(make_ohlcv_dict: Callable[..., dict[Symbol, pd.DataFrame]]) -> OHLCVs:
    dates = pd.date_range("2024-01-01", periods=len(CLOSES), freq="D", tz="UTC")
    return OHLCVs(
        {"1d": make_ohlcv_dict([str(date) for date in dates], [BTC], [CLOSES])}
    )


def _candles_flagged(ohlcvs: OHLCVs, column: str) -> list[int]:
    return np.flatnonzero(ohlcvs.column(BTC, "1d", column)).tolist()


def test_a_crossing_up_signals_buy(make_strategy, ohlcvs):
    strategy = make_strategy()

    strategy.generate_profile_signals(strategy.profiles[0], ohlcvs)

    assert _candles_flagged(ohlcvs, "buy") == CROSSING_UP_CANDLES


def test_a_crossing_down_signals_sell(make_strategy, ohlcvs):
    strategy = make_strategy()

    strategy.generate_profile_signals(strategy.profiles[0], ohlcvs)

    assert _candles_flagged(ohlcvs, "sell") == CROSSING_DOWN_CANDLES


def test_a_long_entry_needs_the_close_above_the_trend(make_strategy, ohlcvs):
    strategy = make_strategy(trend_ma_length=TREND_LENGTH)

    strategy.generate_profile_signals(strategy.profiles[0], ohlcvs)

    assert _candles_flagged(ohlcvs, "long_entry") == LONG_ENTRY_CANDLES


def test_a_short_entry_needs_the_close_below_the_trend(make_strategy, ohlcvs):
    strategy = make_strategy(trend_ma_length=TREND_LENGTH)

    strategy.generate_profile_signals(strategy.profiles[0], ohlcvs)

    assert _candles_flagged(ohlcvs, "short_entry") == SHORT_ENTRY_CANDLES


def test_a_long_exit_ignores_the_trend(make_strategy, ohlcvs):
    strategy = make_strategy(trend_ma_length=TREND_LENGTH)

    strategy.generate_profile_signals(strategy.profiles[0], ohlcvs)

    assert _candles_flagged(ohlcvs, "long_exit") == CROSSING_DOWN_CANDLES


def test_a_short_exit_ignores_the_trend(make_strategy, ohlcvs):
    strategy = make_strategy(trend_ma_length=TREND_LENGTH)

    strategy.generate_profile_signals(strategy.profiles[0], ohlcvs)

    assert _candles_flagged(ohlcvs, "short_exit") == CROSSING_UP_CANDLES
