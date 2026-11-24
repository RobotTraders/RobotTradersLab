"""
Tests for accounting methods implementations.
"""

from datetime import datetime

import pytest

from robottraderslab.analyser.accounting_methods import (
    AccountingMethodChoice,
    FIFOMethod,
    LIFOMethod,
    get_accounting_method,
)


@pytest.fixture
def empty_trades() -> list[dict]:
    """Returns an empty list of trades."""
    return []


@pytest.fixture
def single_trade() -> dict:
    """Returns a sample trade for testing."""
    return {
        "entry_time": datetime(2023, 1, 1),
        "symbol": "BTC/USD",
        "side": "long",
        "quantity": 1.0,
        "entry_price": 30000.0,
        "entry_value": 30000.0,
        "entry_fee": 15.0,
    }


@pytest.fixture
def multiple_trades() -> list[dict]:
    """Returns a list of multiple trades for testing."""
    return [
        {
            "entry_time": datetime(2023, 1, 1),
            "symbol": "BTC/USD",
            "side": "long",
            "quantity": 1.0,
            "entry_price": 30000.0,
            "entry_value": 30000.0,
            "entry_fee": 15.0,
        },
        {
            "entry_time": datetime(2023, 1, 2),
            "symbol": "BTC/USD",
            "side": "long",
            "quantity": 0.5,
            "entry_price": 31000.0,
            "entry_value": 15500.0,
            "entry_fee": 7.75,
        },
    ]


@pytest.fixture
def create_sequenced_trades() -> list[dict]:
    """Create a sequence of three trades with distinct timestamps and prices."""
    return [
        {
            "entry_time": datetime(2023, 1, 1),
            "symbol": "BTC/USD",
            "side": "long",
            "quantity": 1.0,
            "entry_price": 30000.0,
            "entry_value": 30000.0,
            "entry_fee": 15.0,
        },
        {
            "entry_time": datetime(2023, 1, 2),
            "symbol": "BTC/USD",
            "side": "long",
            "quantity": 2.0,
            "entry_price": 32000.0,
            "entry_value": 64000.0,
            "entry_fee": 32.0,
        },
        {
            "entry_time": datetime(2023, 1, 3),
            "symbol": "BTC/USD",
            "side": "long",
            "quantity": 0.5,
            "entry_price": 35000.0,
            "entry_value": 17500.0,
            "entry_fee": 8.75,
        },
    ]


class TestFIFOMethod:
    def test_handle_entry_adds_trade_to_end(self, empty_trades, single_trade):
        method = FIFOMethod()
        result = method.handle_entry(empty_trades, single_trade)

        assert len(result) == 1
        assert result[0] == single_trade

    def test_handle_entry_preserves_existing_trades(
        self, multiple_trades, single_trade
    ):
        method = FIFOMethod()
        initial_count = len(multiple_trades)
        result = method.handle_entry(multiple_trades.copy(), single_trade)

        assert len(result) == initial_count + 1
        assert result[-1] == single_trade
        assert result[0] == multiple_trades[0]

    def test_order_trades_for_exit_returns_same_order(self, multiple_trades):
        method = FIFOMethod()
        result = method.order_trades_for_exit(multiple_trades.copy())

        assert result == multiple_trades

    def test_fifo_processes_oldest_trades_first(self, create_sequenced_trades):
        method = FIFOMethod()
        trades = []

        for trade in create_sequenced_trades:
            trades = method.handle_entry(trades, trade)

        assert (
            trades[0]["entry_time"] < trades[1]["entry_time"] < trades[2]["entry_time"]
        )

        exit_order = method.order_trades_for_exit(trades)

        assert exit_order[0]["entry_price"] == 30000.0
        assert exit_order[0]["entry_time"] == datetime(2023, 1, 1)

        assert exit_order[2]["entry_price"] == 35000.0
        assert exit_order[2]["entry_time"] == datetime(2023, 1, 3)


class TestLIFOMethod:
    def test_handle_entry_adds_trade_to_end(self, empty_trades, single_trade):
        method = LIFOMethod()
        result = method.handle_entry(empty_trades, single_trade)

        assert len(result) == 1
        assert result[0] == single_trade

    def test_handle_entry_preserves_existing_trades(
        self, multiple_trades, single_trade
    ):
        method = LIFOMethod()
        initial_count = len(multiple_trades)
        result = method.handle_entry(multiple_trades.copy(), single_trade)

        assert len(result) == initial_count + 1
        assert result[-1] == single_trade
        assert result[0] == multiple_trades[0]

    def test_order_trades_for_exit_returns_reverse_order(self, multiple_trades):
        method = LIFOMethod()
        result = method.order_trades_for_exit(multiple_trades.copy())

        assert len(result) == len(multiple_trades)
        assert result[0] == multiple_trades[1]
        assert result[1] == multiple_trades[0]

    def test_lifo_processes_newest_trades_first(self, create_sequenced_trades):
        method = LIFOMethod()
        trades = []

        for trade in create_sequenced_trades:
            trades = method.handle_entry(trades, trade)

        assert (
            trades[0]["entry_time"] < trades[1]["entry_time"] < trades[2]["entry_time"]
        )

        exit_order = method.order_trades_for_exit(trades)

        assert exit_order[0]["entry_price"] == 35000.0
        assert exit_order[0]["entry_time"] == datetime(2023, 1, 3)

        assert exit_order[2]["entry_price"] == 30000.0
        assert exit_order[2]["entry_time"] == datetime(2023, 1, 1)


class TestAccountingMethodFactory:
    def test_get_accounting_method_returns_correct_implementation(self):
        fifo_method = get_accounting_method(AccountingMethodChoice.FIFO)
        lifo_method = get_accounting_method(AccountingMethodChoice.LIFO)

        assert isinstance(fifo_method, FIFOMethod)
        assert isinstance(lifo_method, LIFOMethod)

    def test_get_accounting_method_raises_for_unknown_method(self):
        invalid_choice = AccountingMethodChoice.AVERAGE

        with pytest.raises(ValueError, match="Unknown accounting method"):
            get_accounting_method(invalid_choice)
