import sys
from typing import ClassVar

import pytest

from robottraderslab.backtester import run_backtest
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exchanges import MarketType
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies import OHLCVs, StrategyProtocol, StrategyRequirements

_THIS_MODULE = sys.modules[__name__]


@pytest.fixture(autouse=True)
def _register_test_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "test_strategy_market_type", _THIS_MODULE)


@pytest.fixture
def toml_config() -> str:
    return """\
[backtest]
initial_balance = { USDT = 10000.0 }
maker_fee_rate = 0.001
taker_fee_rate = 0.002
start_date = "2024-01-01"
end_date = "2024-01-15"

[backtest.ohlcv_provider]
ohlcv_provider = "mock"

[strategy]
strategy_class = "test_strategy_market_type.{placeholder}"
"""


class _AccountAssertedException(Exception):
    """Raised when the expected account type was wired into the strategy."""


class _AccountAssertingStrategy(StrategyProtocol):
    expected_account_type: ClassVar[type]

    def __init__(self, *, account, **_kwargs) -> None:
        if not isinstance(account, self.expected_account_type):
            raise AssertionError(
                f"Expected {self.expected_account_type.__name__}, got {type(account).__name__}"
            )
        raise _AccountAssertedException()

    async def setup(self, requirements: StrategyRequirements) -> None: ...

    def generate_trading_signals(self, ohlcvs: OHLCVs) -> None: ...

    def book_trading_actions(
        self, ohlcvs, state, timestamp, bookkeeper, triggered_timeframes
    ) -> None: ...


class FuturesDeclaringStrategy(_AccountAssertingStrategy):
    market_type: ClassVar[MarketType] = "futures"
    expected_account_type: ClassVar[type] = FuturesAccount


def test_futures_market_type_wires_futures_account(toml_config):
    bot_config = BotConfig.from_text(
        toml_config.replace("{placeholder}", "FuturesDeclaringStrategy")
    )

    with pytest.raises(_AccountAssertedException):
        run_backtest(bot_config)
