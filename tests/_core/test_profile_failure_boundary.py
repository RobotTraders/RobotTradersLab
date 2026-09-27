import pytest

from robottraderslab._core.profile_failure_boundary import profile_failure_boundary
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
    MissingOhlcvDataError,
    StrategyCriticalError,
)
from robottraderslab.strategies import TradingMode


def _boundary(trading_mode: TradingMode):
    return profile_failure_boundary(trading_mode, "setup-1 (BTC/USDT:USDT)")


class TestLiveMode:
    def test_recoverable_error_is_contained(self):
        with _boundary(TradingMode.LIVE):
            raise ExchangeRecoverableError("rejected")

    def test_internal_error_is_contained(self):
        with _boundary(TradingMode.LIVE):
            raise KeyError("missing column")

    def test_missing_data_is_contained(self):
        with _boundary(TradingMode.LIVE):
            raise MissingOhlcvDataError("BTC/USDT:USDT", "1d")

    def test_exchange_critical_error(self):
        with pytest.raises(ExchangeCriticalError):
            with _boundary(TradingMode.LIVE):
                raise ExchangeCriticalError("auth failed")

    def test_strategy_critical_error(self):
        with pytest.raises(StrategyCriticalError):
            with _boundary(TradingMode.LIVE):
                raise StrategyCriticalError("bad config")


class TestBacktestMode:
    def test_recoverable_error_propagates(self):
        with pytest.raises(ExchangeRecoverableError):
            with _boundary(TradingMode.BACKTEST):
                raise ExchangeRecoverableError("rejected")

    def test_missing_data_propagates(self):
        with pytest.raises(MissingOhlcvDataError):
            with _boundary(TradingMode.BACKTEST):
                raise MissingOhlcvDataError("BTC/USDT:USDT", "1d")

    def test_internal_error_propagates(self):
        with pytest.raises(KeyError):
            with _boundary(TradingMode.BACKTEST):
                raise KeyError("missing column")
