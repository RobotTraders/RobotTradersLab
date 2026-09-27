from unittest.mock import Mock

import pytest

from robottraderslab.exchanges import FuturesExchangeProtocol
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies import TradingMode, TradingSystem


@pytest.fixture
def mock_exchange() -> Mock:
    exchange = Mock(spec=FuturesExchangeProtocol)
    exchange.placement_reserve_rate = 0.0
    return exchange


@pytest.fixture
def mock_account(mock_exchange: Mock) -> FuturesAccount:
    return FuturesAccount(exchange=mock_exchange)


@pytest.fixture
def backtest_trading_system() -> TradingSystem:
    return TradingSystem(trading_mode=TradingMode.BACKTEST)
