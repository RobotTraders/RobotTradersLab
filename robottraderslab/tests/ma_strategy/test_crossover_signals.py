from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from robottraderslab import Symbol
from robottraderslab.futures import FuturesAccount
from robottraderslab.ma_strategy.futures_ma_strategy import FuturesMAStrategy
from robottraderslab.strategies import (
    OHLCVs,
    OrderSide,
    PositionSide,
    TradingSystem,
)

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
FAST_BELOW_SLOW_CANDLES = [6, 7, 8, 12, 13]
FAST_ABOVE_SLOW_CANDLES = [3, 4, 5, 9, 10, 11]
AFTER_THE_CROSS_CANDLE = 7
HELD_QUANTITY = 1.5
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
def ohlcvs(make_dated_ohlcvs: Callable[..., OHLCVs]) -> OHLCVs:
    return make_dated_ohlcvs(BTC, "1d", "D", CLOSES)


@pytest.fixture
def candles_flagged(
    ohlcvs: OHLCVs, candles_where_raised: Callable[..., list[int]]
) -> Callable[[str], list[int]]:
    return lambda column: candles_where_raised(ohlcvs, BTC, "1d", column)


def test_a_crossing_up_signals_buy(make_strategy, ohlcvs, candles_flagged):
    ma_strategy = make_strategy()

    ma_strategy.generate_profile_signals(ma_strategy.profiles[0], ohlcvs)

    assert candles_flagged("buy") == CROSSING_UP_CANDLES


def test_a_crossing_down_signals_sell(make_strategy, ohlcvs, candles_flagged):
    ma_strategy = make_strategy()

    ma_strategy.generate_profile_signals(ma_strategy.profiles[0], ohlcvs)

    assert candles_flagged("sell") == CROSSING_DOWN_CANDLES


def test_a_long_entry_needs_the_close_above_the_trend(
    make_strategy, ohlcvs, candles_flagged
):
    ma_strategy = make_strategy(trend_ma_length=TREND_LENGTH)

    ma_strategy.generate_profile_signals(ma_strategy.profiles[0], ohlcvs)

    assert candles_flagged("long_entry") == LONG_ENTRY_CANDLES


def test_a_short_entry_needs_the_close_below_the_trend(
    make_strategy, ohlcvs, candles_flagged
):
    ma_strategy = make_strategy(trend_ma_length=TREND_LENGTH)

    ma_strategy.generate_profile_signals(ma_strategy.profiles[0], ohlcvs)

    assert candles_flagged("short_entry") == SHORT_ENTRY_CANDLES


@pytest.mark.parametrize(
    "profile_fields",
    [{}, {"trend_ma_length": TREND_LENGTH}],
    ids=["without a trend", "with a trend"],
)
def test_a_long_exit_stays_raised_while_the_fast_is_below_the_slow(
    make_strategy, ohlcvs, candles_flagged, profile_fields
):
    ma_strategy = make_strategy(**profile_fields)

    ma_strategy.generate_profile_signals(ma_strategy.profiles[0], ohlcvs)

    assert candles_flagged("long_exit") == FAST_BELOW_SLOW_CANDLES


@pytest.mark.parametrize(
    "profile_fields",
    [{}, {"trend_ma_length": TREND_LENGTH}],
    ids=["without a trend", "with a trend"],
)
def test_a_short_exit_stays_raised_while_the_fast_is_above_the_slow(
    make_strategy, ohlcvs, candles_flagged, profile_fields
):
    ma_strategy = make_strategy(**profile_fields)

    ma_strategy.generate_profile_signals(ma_strategy.profiles[0], ohlcvs)

    assert candles_flagged("short_exit") == FAST_ABOVE_SLOW_CANDLES


def test_a_held_long_is_exited_on_a_candle_after_the_cross(
    make_strategy, ohlcvs, make_booked_order, hold_position
):
    ma_strategy = make_strategy()
    hold_position(ma_strategy, PositionSide.LONG, HELD_QUANTITY)
    ma_strategy.generate_trading_signals(ohlcvs)
    next(ohlcvs._iter_timeframes(start_idx=AFTER_THE_CROSS_CANDLE))

    exit_order = make_booked_order(ma_strategy, ohlcvs, "2024-01-08 00:00:00")

    assert (exit_order.side, exit_order.quantity) == (OrderSide.SELL, HELD_QUANTITY)
