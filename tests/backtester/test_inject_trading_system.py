import sys

import pytest

from robottraderslab.backtester import run_backtest
from robottraderslab.bootstrap import BotConfig
from robottraderslab.live.runners import run_livebot
from robottraderslab.strategies import (
    OHLCVs,
    StrategyProtocol,
    StrategyRequirements,
    TradingSystem,
)

_THIS_MODULE = sys.modules[__name__]


@pytest.fixture(autouse=True)
def _register_test_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "test_inject_trading_system", _THIS_MODULE)


@pytest.fixture
def toml_config() -> str:
    return """\
[backtest]
initial_balance = { USDT = 10000.0 }
maker_fee_rate = 0.001
taker_fee_rate = 0.002
start_date = "2024-01-01"
end_date = "2025-01-01"

[backtest.ohlcv_provider]
ohlcv_provider = "mock"

[live.ohlcv_provider]
ohlcv_provider = "mock"

[live.trading_account]
exchange = "simulator"
name = "account"
initial_balance = { USDT = 10000.0 }
maker_fee_rate = 0.001
taker_fee_rate = 0.002

[strategy]
strategy_class = "test_inject_trading_system.{placeholder}"
"""


class StrategyLoadRaisesException(Exception):
    """Exception raised when strategy.setup() is called to verify it was invoked."""


class StrategyFixture(StrategyProtocol):
    market_type = "futures"

    async def setup(self, requirements: StrategyRequirements) -> None:
        raise StrategyLoadRaisesException()

    def generate_trading_signals(self, ohlcvs: OHLCVs) -> None:
        pass

    def book_trading_actions(
        self, ohlcvs, state, timestamp, bookkeeper, triggered_timeframes
    ) -> None:
        pass


class StrategyNotUsingTradingSystem(StrategyFixture):
    def __init__(
        self,
        **kwargs,
    ) -> None: ...


class StrategyBacktestOnly(StrategyFixture):
    def __init__(
        self,
        trading_system: TradingSystem,
        **kwargs,
    ) -> None:
        assert trading_system.trading_mode == "backtest"


class StrategyLiveOnly(StrategyFixture):
    def __init__(
        self,
        trading_system: TradingSystem,
        **kwargs,
    ) -> None:
        assert trading_system.trading_mode == "live"


def test_strategy_not_using_trading_system(toml_config):
    bot_config = BotConfig.from_text(
        toml_config.replace("{placeholder}", "StrategyNotUsingTradingSystem")
    )

    with pytest.raises(StrategyLoadRaisesException):
        run_backtest(bot_config)


def test_run_backtest_injects_trading_system(toml_config):
    bot_config = BotConfig.from_text(
        toml_config.replace("{placeholder}", "StrategyBacktestOnly")
    )

    with pytest.raises(StrategyLoadRaisesException):
        run_backtest(bot_config)


def test_run_livebot_injects_trading_system(toml_config):
    bot_config = BotConfig.from_text(
        toml_config.replace("{placeholder}", "StrategyLiveOnly")
    )

    with pytest.raises(StrategyLoadRaisesException):
        run_livebot(bot_config)
