import asyncio
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import (
    FuturesExchangeProtocol,
    MarginMode,
    MarginSettings,
)
from robottraderslab.futures import FuturesAccount
from robottraderslab.futures.futures_set_leverage import SetLeverageAction
from robottraderslab.futures.futures_set_margin_mode import SetMarginModeAction
from robottraderslab.ma_strategy.futures_ma_strategy import FuturesMAStrategy
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    BookKeeper,
    OHLCVs,
    StrategyRequirements,
    TradingMode,
    TradingSystem,
)


@pytest.fixture
def mock_exchange() -> MagicMock:
    exchange = MagicMock(spec=FuturesExchangeProtocol)
    exchange.get_balances.return_value = {}
    exchange.get_open_positions.return_value = {}
    exchange.get_equity.return_value = 0.0
    exchange.get_open_orders.return_value = []
    return exchange


@pytest.fixture
def mock_account(mock_exchange) -> FuturesAccount:
    return FuturesAccount(exchange=mock_exchange)


@pytest.fixture
def backtest_trading_system() -> TradingSystem:
    return TradingSystem(trading_mode=TradingMode.BACKTEST)


@pytest.fixture
def btc_symbol() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


class TestMarginTargets:
    @pytest.fixture
    def strategy_single_profile(
        self, mock_account, backtest_trading_system
    ) -> FuturesMAStrategy:
        profiles: list[dict[str, Any]] = [
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1d",
                "fast_ma_length": 7,
                "slow_ma_length": 20,
                "available_balance_ratio": 0.25,
                "leverage": 5.0,
                "margin_mode": "cross",
            }
        ]
        strategy = FuturesMAStrategy(
            account=mock_account,
            trading_system=backtest_trading_system,
            config_dir=Path("bot-config-dir"),
            profiles=profiles,
        )
        strategy._position_tracker = MagicMock()
        strategy._balances = {}
        return strategy

    @pytest.fixture
    def strategy_multiple_profiles(
        self, mock_account, backtest_trading_system
    ) -> FuturesMAStrategy:
        profiles: list[dict[str, Any]] = [
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "1d",
                "fast_ma_length": 7,
                "slow_ma_length": 20,
                "available_balance_ratio": 0.25,
                "leverage": 10.0,
                "margin_mode": "isolated",
                "tag": "slow",
            },
            {
                "symbol": "ETH/USDT:USDT",
                "timeframe": "1d",
                "fast_ma_length": 5,
                "slow_ma_length": 15,
                "available_balance_ratio": 0.25,
                "leverage": 3.0,
                "margin_mode": "cross",
            },
            {
                "symbol": "BTC/USDT:USDT",
                "timeframe": "4h",
                "fast_ma_length": 3,
                "slow_ma_length": 7,
                "available_balance_ratio": 0.25,
                "leverage": 20.0,
                "margin_mode": "cross",
                "tag": "fast",
            },
        ]
        strategy = FuturesMAStrategy(
            account=mock_account,
            trading_system=backtest_trading_system,
            config_dir=Path("bot-config-dir"),
            profiles=profiles,
        )
        strategy._position_tracker = MagicMock()
        strategy._balances = {}
        return strategy

    def _declared_targets(
        self, strategy: FuturesMAStrategy
    ) -> Mapping[Symbol, MarginSettings]:
        requirements = StrategyRequirements()
        asyncio.run(strategy.setup(requirements))
        return requirements.account._get_all()[0].margin_targets

    def test_a_profile_declares_its_settings(self, strategy_single_profile, btc_symbol):
        targets = self._declared_targets(strategy_single_profile)

        assert targets == {
            btc_symbol: MarginSettings(leverage=5.0, margin_mode=MarginMode.CROSS)
        }

    def test_first_profile_settings_win_for_duplicate_symbol(
        self, strategy_multiple_profiles, btc_symbol
    ):
        targets = self._declared_targets(strategy_multiple_profiles)

        assert targets == {
            btc_symbol: MarginSettings(leverage=10.0, margin_mode=MarginMode.ISOLATED),
            Symbol.create("ETH/USDT:USDT"): MarginSettings(
                leverage=3.0, margin_mode=MarginMode.CROSS
            ),
        }

    def test_a_drifted_symbol_books_no_margin_action(self, strategy_single_profile):
        bookkeeper = BookKeeper()
        ohlcvs = Mock(spec=OHLCVs)
        ohlcvs.current.return_value = math.nan
        account_name = strategy_single_profile.account.name
        snapshots = AccountSnapshots(
            {
                account_name: AccountSnapshot(
                    account_name=account_name,
                    positions={},
                    margin_settings={
                        profile.symbol: MarginSettings(
                            leverage=None, margin_mode=MarginMode.ISOLATED
                        )
                        for profile in strategy_single_profile.profiles
                    },
                )
            }
        )

        strategy_single_profile.book_trading_actions(
            ohlcvs,
            snapshots,
            datetime.fromisoformat("2024-01-01 12:00:00"),
            bookkeeper,
            {"1d", "4h"},
        )

        setters = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, (SetLeverageAction, SetMarginModeAction))
        ]
        assert setters == []
