import asyncio
from datetime import datetime

import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    Balance,
    BookKeeper,
    OHLCVs,
    StrategyRequirements,
    TradingMode,
)
from robottraderslab.strategies.futures import (
    FuturesMarketOrderAction,
    MarginMode,
    MarginSettings,
)
from robottraderslab_impulse import ImpulseStrategy

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")
ACCOUNT_NAME = "test"
TIMEFRAME = "1d"
TIMESTAMP = datetime(2024, 2, 19)

BTC_PROFILE = {
    "symbol": "BTC/USDT:USDT",
    "timeframe": TIMEFRAME,
    "trix_length": 9,
    "signal_length": 21,
    "trend_length": 200,
    "total_balance_ratio": 0.25,
    "leverage": 5.0,
    "margin_mode": "isolated",
}
ETH_PROFILE = {**BTC_PROFILE, "symbol": "ETH/USDT:USDT"}


def _stand_on_first_candle(ohlcvs: OHLCVs) -> None:
    """Set the iteration cursor so `current` and `signal` read the first candle."""
    next(ohlcvs._iter_timeframes())


def _make_ohlcvs(symbols_with_signals: list[Symbol]) -> OHLCVs:
    dates = pd.date_range("2024-02-19", periods=1, freq="1D")
    frames = {
        symbol: pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [1000.0],
            },
            index=dates,
        )
        for symbol in (BTC, ETH)
    }
    ohlcvs = OHLCVs({TIMEFRAME: frames})

    for symbol in symbols_with_signals:
        profile_id = f"{symbol}@{TIMEFRAME}"
        ohlcvs.add_column(symbol, TIMEFRAME, f"{profile_id}_long_entry", [1.0])
        ohlcvs.add_column(symbol, TIMEFRAME, f"{profile_id}_long_exit", [0.0])
        ohlcvs.add_column(symbol, TIMEFRAME, f"{profile_id}_short_entry", [0.0])
        ohlcvs.add_column(symbol, TIMEFRAME, f"{profile_id}_short_exit", [0.0])

    _stand_on_first_candle(ohlcvs)

    return ohlcvs


def _account_state() -> AccountSnapshot:
    snapshot = AccountSnapshot(
        account_name=ACCOUNT_NAME,
        balances={"USDT": Balance(locked=0.0, total=10_000.0)},
        positions={},
        open_orders=[],
        margin_settings={
            symbol: MarginSettings(leverage=None, margin_mode=MarginMode.CROSS)
            for symbol in (BTC, ETH)
        },
        conversion_rates={symbol: 1.0 for symbol in (BTC, ETH)},
    )
    return AccountSnapshots({ACCOUNT_NAME: snapshot})


def _entry_orders_for(bookkeeper: BookKeeper, symbol: Symbol) -> list:
    return [
        action
        for action in bookkeeper.list_actions()
        if isinstance(action, FuturesMarketOrderAction) and action.symbol == symbol
    ]


def _load(strategy: ImpulseStrategy) -> ImpulseStrategy:
    asyncio.run(strategy.setup(StrategyRequirements()))
    return strategy


def test_live_mode_with_internal_error_in_one_profile(make_strategy):
    strategy = _load(
        make_strategy([BTC_PROFILE, ETH_PROFILE], trading_mode=TradingMode.LIVE)
    )
    bookkeeper = BookKeeper()

    strategy.book_trading_actions(
        _make_ohlcvs([ETH]), _account_state(), TIMESTAMP, bookkeeper, {TIMEFRAME}
    )

    assert not _entry_orders_for(bookkeeper, BTC)
    assert _entry_orders_for(bookkeeper, ETH)


def test_backtest_mode_with_internal_error_in_one_profile(make_strategy):
    strategy = _load(make_strategy([BTC_PROFILE, ETH_PROFILE]))

    with pytest.raises(KeyError):
        strategy.book_trading_actions(
            _make_ohlcvs([ETH]), _account_state(), TIMESTAMP, BookKeeper(), {TIMEFRAME}
        )
