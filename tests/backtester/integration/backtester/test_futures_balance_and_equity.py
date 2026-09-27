import io
from collections.abc import Callable
from datetime import datetime

import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FeeRates,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
)
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies import BookKeeper

BTCUSDT_PERP = Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def simulated_exchange():
    return SimulatedFuturesExchange.create_from_settings(
        initial_balance={"USDT": 10_000.0},
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        fee_mode="cost",
    )


@pytest.fixture
def simulation_engine(
    simulated_exchange: SimulatedFuturesExchange,
) -> FuturesSimulationEngine:
    return simulated_exchange.simulation_engine


@pytest.fixture
def fill_recorder(
    simulated_exchange: SimulatedFuturesExchange,
) -> CacheFillRecorder:
    return simulated_exchange.fill_recorder


@pytest.fixture
def account(simulated_exchange):
    return FuturesAccount(simulated_exchange)


class TestFuturesBalanceAndEquity:
    """Test class for position-based balance and equity tests with shared helper methods."""

    @pytest.fixture(autouse=True)
    def setup(
        self, account, simulation_engine, fill_recorder, make_strategy, run_backtest
    ):
        self.account: FuturesAccount = account
        self.simulation_engine: FuturesSimulationEngine = simulation_engine
        self.fill_recorder: CacheFillRecorder = fill_recorder
        self.make_strategy: Callable = make_strategy
        self.run_backtest: Callable = run_backtest

    def test_no_trades(self, csv_3_daily_no_price_fluctuation):
        """Test with no trades - balance and equity should remain unchanged."""
        self.assert_balances_and_equity(
            expected_available=10_000.0,
            expected_locked=0.0,
            expected_total=10_000.0,
            expected_equity=10_000.0,
        )

        self.run_backtest_with_no_trades(csv_3_daily_no_price_fluctuation)

        self.assert_balances_and_equity(
            expected_available=10_000.0,
            expected_locked=0.0,
            expected_total=10_000.0,
            expected_equity=10_000.0,
        )

    def test_with_positions_no_price_fluctuation(
        self, csv_3_daily_no_price_fluctuation
    ):
        self.run_backtest_with_position(
            csv_3_daily_no_price_fluctuation, FeeRates(0.001, 0.001)
        )

        self.assert_balances_and_equity(
            expected_available=9900.0,
            expected_locked=100.0 - 0.1,
            expected_total=10_000.0 - 0.1,
            expected_equity=10_000.0 - 0.1,
        )

    def test_with_positive_unrealised_PnL(self, csv_3_daily_with_price_increases):
        self.run_backtest_with_position(csv_3_daily_with_price_increases)
        self.assert_balances_and_equity(
            expected_available=9890.0,
            expected_locked=110.0,
            expected_total=10_000.0,
            expected_equity=10_020.0,
        )

    def test_with_negative_unrealised_PnL(self, csv_3_daily_with_price_decreases):
        self.run_backtest_with_position(csv_3_daily_with_price_decreases)
        self.assert_balances_and_equity(
            expected_available=9910.0,
            expected_locked=90.0,
            expected_total=10_000.0,
            expected_equity=9980.0,
        )

    def test_with_position_entry_and_exit_no_price_fluctuation(
        self, csv_3_daily_no_price_fluctuation
    ):
        self.run_backtest_with_position_and_exit(csv_3_daily_no_price_fluctuation)
        self.assert_balances_and_equity(
            expected_available=10_000.0,
            expected_locked=0.0,
            expected_total=10_000.0,
            expected_equity=10_000.0,
        )

    def test_with_positive_realised_PnL(self, csv_3_daily_with_price_increases):
        self.run_backtest_with_position_and_exit(csv_3_daily_with_price_increases)
        self.assert_balances_and_equity(
            expected_available=10_010.0,
            expected_locked=0.0,
            expected_total=10_010.0,
            expected_equity=10_010.0,
        )

    def test_with_negative_realised_PnL(self, csv_3_daily_with_price_decreases):
        self.run_backtest_with_position_and_exit(csv_3_daily_with_price_decreases)
        self.assert_balances_and_equity(
            expected_available=9990.0,
            expected_locked=0.0,
            expected_total=9990.0,
            expected_equity=9990.0,
        )

    def run_backtest_with_position(
        self, csv_ohlcvs: io.StringIO, fee_rates=FeeRates(0.0, 0.0)
    ) -> None:
        self.simulation_engine._fee_rates = fee_rates

        def take_position(
            count: int,
            timestamp: datetime,
            bookkeeper: BookKeeper,
            triggered_timeframes: set[TimeFrame],
        ) -> None:
            match count:
                case 0:
                    action = self.account.long_entry(BTCUSDT_PERP, 1.0).build()
                    bookkeeper.add(action)

        strategy = self.make_strategy(csv_ohlcvs, BTCUSDT_PERP, take_position)
        self.run_backtest(strategy, self.simulation_engine, self.fill_recorder)
        assert len(self.simulation_engine.open_positions) == 1

    def run_backtest_with_position_and_exit(self, csv_ohlcvs: io.StringIO) -> None:
        def take_and_exit_position(
            count: int,
            timestamp: datetime,
            bookkeeper: BookKeeper,
            triggered_timeframes: set[TimeFrame],
        ) -> None:
            match count:
                case 0:
                    action = self.account.long_entry(BTCUSDT_PERP, 1.0).build()
                    bookkeeper.add(action)
                case 1:
                    action = self.account.long_exit(BTCUSDT_PERP, 1.0).build()
                    bookkeeper.add(action)

        strategy = self.make_strategy(csv_ohlcvs, BTCUSDT_PERP, take_and_exit_position)
        self.run_backtest(strategy, self.simulation_engine, self.fill_recorder)
        assert len(self.simulation_engine.open_positions) == 0

    def run_backtest_with_no_trades(self, csv_ohlcvs: io.StringIO) -> None:
        def no_trades(
            count: int,
            timestamp: datetime,
            bookkeeper: BookKeeper,
            triggered_timeframes: set[TimeFrame],
        ) -> None:
            pass

        strategy = self.make_strategy(csv_ohlcvs, BTCUSDT_PERP, no_trades)
        self.run_backtest(strategy, self.simulation_engine, self.fill_recorder)

    def assert_balances_and_equity(
        self,
        expected_available: float,
        expected_locked: float,
        expected_total: float,
        expected_equity: float,
    ) -> None:
        balances = self.simulation_engine.get_balances()
        usdt = balances["USDT"]
        assert len(balances) == 1
        assert usdt.available == expected_available
        assert usdt.locked == expected_locked
        assert usdt.total == expected_total
        assert self.simulation_engine.get_equity("USDT") == expected_equity
