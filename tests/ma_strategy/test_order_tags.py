from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.futures import FuturesAccount, FuturesMarketOrderAction
from robottraderslab.ma_strategy.futures_ma_strategy import FuturesMAStrategy
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    Balance,
    BookKeeper,
    OHLCVs,
    PositionSide,
    PositionTracker,
    StrategyRequirements,
    TrackedPosition,
    TradingSystem,
    tag_of,
)
from robottraderslab.strategies.futures import MarginSettings

_MARGIN_CURRENCY = "USDT"


def _make_strategy(
    mock_account: FuturesAccount,
    backtest_trading_system: TradingSystem,
    profiles: list[dict[str, Any]],
) -> FuturesMAStrategy:
    strategy = FuturesMAStrategy(
        account=mock_account,
        trading_system=backtest_trading_system,
        config_dir=Path("bot-config-dir"),
        profiles=profiles,
    )
    strategy._position_tracker = Mock(spec=PositionTracker)
    strategy._position_tracker.get.return_value = None
    return strategy


def _ohlcvs_signalling(wanted_column: str) -> OHLCVs:
    ohlcvs = Mock(spec=OHLCVs)
    ohlcvs.current.return_value = 100.0
    ohlcvs.signal.side_effect = lambda symbol, timeframe, name: name == wanted_column
    return ohlcvs


def _book(strategy: FuturesMAStrategy, ohlcvs: OHLCVs) -> BookKeeper:
    bookkeeper = BookKeeper()
    account_name = strategy.account.name
    current_settings = {
        profile.symbol: MarginSettings(
            leverage=profile.leverage, margin_mode=profile.margin_mode
        )
        for profile in strategy.profiles
    }
    snapshots = AccountSnapshots(
        {
            account_name: AccountSnapshot(
                account_name=account_name,
                positions={},
                balances={_MARGIN_CURRENCY: Balance(locked=0.0, total=100_000.0)},
                margin_settings=current_settings,
            )
        }
    )
    strategy.book_trading_actions(
        ohlcvs,
        snapshots,
        datetime.fromisoformat("2024-01-01 12:00:00"),
        bookkeeper,
        [strategy.profiles[0].timeframe],
    )
    return bookkeeper


def _only_order_action(bookkeeper: BookKeeper) -> FuturesMarketOrderAction:
    order_actions = [
        action
        for action in bookkeeper.list_actions()
        if isinstance(action, FuturesMarketOrderAction)
    ]
    assert len(order_actions) == 1
    return order_actions[0]


class TestCrossoverTags:
    def test_entry_long_uses_the_timeframe_without_tag(
        self, mock_account, backtest_trading_system
    ):
        profile = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1d",
            "fast_ma_length": 7,
            "slow_ma_length": 20,
            "available_balance_ratio": 0.25,
        }
        strategy = _make_strategy(mock_account, backtest_trading_system, [profile])
        ohlcvs = _ohlcvs_signalling("buy")

        bookkeeper = _book(strategy, ohlcvs)

        order = _only_order_action(bookkeeper)
        assert tag_of(order.client_order_id) == "1d"

    def test_entry_short_composes_the_timeframe_with_the_profiles_tag(
        self, mock_account, backtest_trading_system
    ):
        profile = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1d",
            "fast_ma_length": 7,
            "slow_ma_length": 20,
            "available_balance_ratio": 0.25,
            "tag": "slow",
        }
        strategy = _make_strategy(mock_account, backtest_trading_system, [profile])
        ohlcvs = _ohlcvs_signalling("slow_sell")

        bookkeeper = _book(strategy, ohlcvs)

        order = _only_order_action(bookkeeper)
        assert tag_of(order.client_order_id) == "1d-slow"

    def test_exit_long_composes_the_timeframe_with_the_profiles_tag(
        self, mock_account, backtest_trading_system
    ):
        profile = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1d",
            "fast_ma_length": 7,
            "slow_ma_length": 20,
            "available_balance_ratio": 0.25,
            "tag": "slow",
        }
        strategy = _make_strategy(mock_account, backtest_trading_system, [profile])
        strategy._position_tracker.get.return_value = TrackedPosition(
            side=PositionSide.LONG, quantity=1.5
        )
        ohlcvs = _ohlcvs_signalling("slow_sell")

        bookkeeper = _book(strategy, ohlcvs)

        order = _only_order_action(bookkeeper)
        assert tag_of(order.client_order_id) == "1d-slow"

    def test_exit_short_uses_the_timeframe_without_tag(
        self, mock_account, backtest_trading_system
    ):
        profile = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1d",
            "fast_ma_length": 7,
            "slow_ma_length": 20,
            "available_balance_ratio": 0.25,
        }
        strategy = _make_strategy(mock_account, backtest_trading_system, [profile])
        strategy._position_tracker.get.return_value = TrackedPosition(
            side=PositionSide.SHORT, quantity=1.5
        )
        ohlcvs = _ohlcvs_signalling("buy")

        bookkeeper = _book(strategy, ohlcvs)

        order = _only_order_action(bookkeeper)
        assert tag_of(order.client_order_id) == "1d"


class TestLongShortTags:
    def test_entry_long_composes_the_timeframe_with_the_profiles_tag(
        self, mock_account, backtest_trading_system
    ):
        profile = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1d",
            "fast_ma_length": 7,
            "slow_ma_length": 20,
            "trend_ma_length": 200,
            "available_balance_ratio": 0.25,
            "tag": "trend",
        }
        strategy = _make_strategy(mock_account, backtest_trading_system, [profile])
        ohlcvs = _ohlcvs_signalling("trend_long_entry")

        bookkeeper = _book(strategy, ohlcvs)

        order = _only_order_action(bookkeeper)
        assert tag_of(order.client_order_id) == "1d-trend"

    def test_entry_short_uses_the_timeframe_without_tag(
        self, mock_account, backtest_trading_system
    ):
        profile = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1d",
            "fast_ma_length": 7,
            "slow_ma_length": 20,
            "trend_ma_length": 200,
            "available_balance_ratio": 0.25,
        }
        strategy = _make_strategy(mock_account, backtest_trading_system, [profile])
        ohlcvs = _ohlcvs_signalling("short_entry")

        bookkeeper = _book(strategy, ohlcvs)

        order = _only_order_action(bookkeeper)
        assert tag_of(order.client_order_id) == "1d"

    def test_exit_long_uses_the_timeframe_without_tag(
        self, mock_account, backtest_trading_system
    ):
        profile = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1d",
            "fast_ma_length": 7,
            "slow_ma_length": 20,
            "trend_ma_length": 200,
            "available_balance_ratio": 0.25,
        }
        strategy = _make_strategy(mock_account, backtest_trading_system, [profile])
        strategy._position_tracker.get.return_value = TrackedPosition(
            side=PositionSide.LONG, quantity=1.5
        )
        ohlcvs = _ohlcvs_signalling("long_exit")

        bookkeeper = _book(strategy, ohlcvs)

        order = _only_order_action(bookkeeper)
        assert tag_of(order.client_order_id) == "1d"

    def test_exit_short_composes_the_timeframe_with_the_profiles_tag(
        self, mock_account, backtest_trading_system
    ):
        profile = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1d",
            "fast_ma_length": 7,
            "slow_ma_length": 20,
            "trend_ma_length": 200,
            "available_balance_ratio": 0.25,
            "tag": "trend",
        }
        strategy = _make_strategy(mock_account, backtest_trading_system, [profile])
        strategy._position_tracker.get.return_value = TrackedPosition(
            side=PositionSide.SHORT, quantity=1.5
        )
        ohlcvs = _ohlcvs_signalling("trend_short_exit")

        bookkeeper = _book(strategy, ohlcvs)

        order = _only_order_action(bookkeeper)
        assert tag_of(order.client_order_id) == "1d-trend"


class TestTagValidation:
    async def test_a_custom_tag_at_the_budget_is_accepted(
        self, mock_account, backtest_trading_system
    ):
        profile = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1d",
            "fast_ma_length": 7,
            "slow_ma_length": 20,
            "available_balance_ratio": 0.25,
            "tag": "abcdefgh",
        }
        strategy = _make_strategy(mock_account, backtest_trading_system, [profile])

        await strategy.setup(StrategyRequirements())

    async def test_colliding_tags_on_the_same_symbol_and_timeframe_are_refused(
        self, mock_account, backtest_trading_system
    ):
        profiles = [
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1d",
                "fast_ma_length": 7,
                "slow_ma_length": 20,
                "available_balance_ratio": 0.25,
                "tag": "dup",
            },
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1d",
                "fast_ma_length": 5,
                "slow_ma_length": 15,
                "available_balance_ratio": 0.25,
                "tag": "dup",
            },
        ]
        strategy = _make_strategy(mock_account, backtest_trading_system, profiles)

        with pytest.raises(StrategyCriticalError, match="1d-dup"):
            await strategy.setup(StrategyRequirements())

    async def test_same_tag_on_different_timeframes_is_not_a_collision(
        self, mock_account, backtest_trading_system
    ):
        profiles = [
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1h",
                "fast_ma_length": 7,
                "slow_ma_length": 20,
                "available_balance_ratio": 0.25,
                "tag": "alpha",
            },
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "2h",
                "fast_ma_length": 5,
                "slow_ma_length": 15,
                "available_balance_ratio": 0.25,
                "tag": "alpha",
            },
        ]
        strategy = _make_strategy(mock_account, backtest_trading_system, profiles)

        await strategy.setup(StrategyRequirements())

    async def test_the_same_tag_on_different_symbols_is_not_a_collision(
        self, mock_account, backtest_trading_system
    ):
        profiles = [
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1d",
                "fast_ma_length": 7,
                "slow_ma_length": 20,
                "available_balance_ratio": 0.25,
            },
            {
                "symbol": "ETH/USDT:USDT",
                "timeframe": "1d",
                "fast_ma_length": 5,
                "slow_ma_length": 15,
                "available_balance_ratio": 0.25,
            },
        ]
        strategy = _make_strategy(mock_account, backtest_trading_system, profiles)

        await strategy.setup(StrategyRequirements())
