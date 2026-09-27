import pytest

from robottraderslab import Symbol
from robottraderslab._core import OHLCVRequirement
from robottraderslab.strategies import StrategyRequirements


@pytest.fixture
def symbol() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def another_symbol() -> Symbol:
    return Symbol.create("ETH/USDT:USDT")


class TestOHLCVRequirement:
    def test_creates_with_required_fields(self, symbol):
        requirement = OHLCVRequirement(symbol=symbol, timeframe="4h")

        assert requirement.symbol == symbol
        assert requirement.timeframe == "4h"
        assert requirement.lookback == 0

    def test_creates_with_lookback(self, symbol):
        requirement = OHLCVRequirement(symbol=symbol, timeframe="1d", lookback=100)

        assert requirement.lookback == 100


class TestStrategyRequirements:
    def test_get_all_when_no_requirements(self):
        requirements = StrategyRequirements()

        assert requirements.ohlcv._get_all() == []

    def test_add_stores_requirement(self, symbol):
        requirements = StrategyRequirements()

        requirements.ohlcv.add(symbol, "4h", lookback=50)

        stored = requirements.ohlcv._get_all()
        assert len(stored) == 1
        assert stored[0].symbol == symbol
        assert stored[0].timeframe == "4h"
        assert stored[0].lookback == 50

    def test_add_multiple_requirements(self, symbol, another_symbol):
        requirements = StrategyRequirements()

        requirements.ohlcv.add(symbol, "4h")
        requirements.ohlcv.add(another_symbol, "1d", lookback=200)

        stored = requirements.ohlcv._get_all()
        assert len(stored) == 2

    def test_get_all_returns_copy(self, symbol):
        requirements = StrategyRequirements()
        requirements.ohlcv.add(symbol, "4h")

        first_call = requirements.ohlcv._get_all()
        first_call.clear()

        assert len(requirements.ohlcv._get_all()) == 1
