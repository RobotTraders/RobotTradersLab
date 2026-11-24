from collections.abc import Callable
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies import TradingMode, TradingSystem

BTC_USDT = Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def mock_exchange(make_venue: Callable[..., Mock]) -> Mock:
    exchange = make_venue()
    exchange.placement_reserve_rate = 0.0
    return exchange


@pytest.fixture
async def mock_account(mock_exchange: Mock) -> FuturesAccount:
    account = FuturesAccount(exchange=mock_exchange)
    await account._read_instruments([BTC_USDT])
    return account


@pytest.fixture
def backtest_trading_system() -> TradingSystem:
    return TradingSystem(trading_mode=TradingMode.BACKTEST)
