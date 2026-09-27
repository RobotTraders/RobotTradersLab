from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from robottraderslab import Symbol
from robottraderslab._core import attribute_fill_effects
from robottraderslab.exchanges import (
    Execution,
    OrderSide,
    PositionSide,
    PositionSnapshot,
)

BTCUSDT = Symbol.create("BTC/USDT:USDT")
ETHUSDT = Symbol.create("ETH/USDT:USDT")


def _long(quantity: float, symbol: Symbol = BTCUSDT) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        side=PositionSide.LONG,
        quantity=quantity,
        average_entry_price=100.0,
        entry_time=datetime(2026, 5, 1, tzinfo=UTC),
        leverage=1.0,
        liquidation_price=1.0,
    )


def _short(quantity: float, symbol: Symbol = BTCUSDT) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        side=PositionSide.SHORT,
        quantity=quantity,
        average_entry_price=100.0,
        entry_time=datetime(2026, 5, 1, tzinfo=UTC),
        leverage=1.0,
        liquidation_price=1.0,
    )


def _positions(*held: PositionSnapshot) -> dict[Symbol, PositionSnapshot]:
    return {position.symbol: position for position in held}


@pytest.fixture
def create_execution() -> Callable[..., Execution]:
    def _create_execution(
        execution_id: str,
        side: OrderSide,
        quantity: float,
        minute: int,
        symbol: Symbol = BTCUSDT,
    ) -> Execution:
        return Execution(
            execution_id=execution_id,
            order_id=execution_id,
            symbol=symbol,
            side=side,
            price=100.0,
            quantity=quantity,
            timestamp=datetime(2026, 5, 1, 12, minute, tzinfo=UTC),
        )

    return _create_execution


@pytest.mark.parametrize(
    ("booked", "held", "expected"),
    [
        ([(OrderSide.BUY, 1.0)], _positions(_long(1.0)), ["open"]),
        ([(OrderSide.SELL, 1.0)], _positions(_short(1.0)), ["open"]),
        (
            [(OrderSide.BUY, 1.0), (OrderSide.BUY, 1.0)],
            _positions(_long(2.0)),
            ["open", "increase"],
        ),
        (
            [(OrderSide.SELL, 0.3), (OrderSide.SELL, 0.7)],
            _positions(_short(1.0)),
            ["open", "increase"],
        ),
        (
            [(OrderSide.BUY, 2.0), (OrderSide.SELL, 1.0)],
            _positions(_long(1.0)),
            ["open", "reduce"],
        ),
        (
            [(OrderSide.BUY, 1.0), (OrderSide.SELL, 1.0)],
            _positions(),
            ["open", "close"],
        ),
        (
            [(OrderSide.BUY, 1.0), (OrderSide.SELL, 3.0)],
            _positions(_short(2.0)),
            ["open", "close"],
        ),
        ([(OrderSide.BUY, 0.0)], _positions(), [None]),
    ],
    ids=[
        "long_from_flat",
        "short_from_flat",
        "long_grown",
        "short_grown",
        "long_partly_taken_off",
        "long_flattened",
        "long_flipped_short",
        "nothing_moved",
    ],
)
def test_effect_of_each_move(create_execution, booked, held, expected):
    stream = [
        create_execution(f"e{index}", side, quantity, index)
        for index, (side, quantity) in enumerate(booked)
    ]

    attributed = attribute_fill_effects(stream, held)

    assert [execution.effect for execution in attributed] == expected


def test_position_closed_reopened_and_closed_inside_one_candle(create_execution):
    stream = [
        create_execution("take-profit", OrderSide.SELL, 3.0, 13),
        create_execution("rung-1", OrderSide.BUY, 0.5, 14),
        create_execution("rung-2", OrderSide.BUY, 0.5, 15),
        create_execution("rung-3", OrderSide.BUY, 1.0, 16),
        create_execution("exit", OrderSide.SELL, 2.0, 17),
    ]

    attributed = attribute_fill_effects(stream, _positions())

    assert {execution.execution_id: execution.effect for execution in attributed} == {
        "take-profit": "close",
        "rung-1": "open",
        "rung-2": "increase",
        "rung-3": "increase",
        "exit": "close",
    }


def test_stream_arriving_newest_first(create_execution):
    stream = [
        create_execution("exit", OrderSide.SELL, 1.0, 1),
        create_execution("entry", OrderSide.BUY, 1.0, 0),
    ]

    attributed = attribute_fill_effects(stream, _positions())

    assert [execution.effect for execution in attributed] == ["close", "open"]


def test_two_exits_sharing_an_instant_close_on_the_later_one(create_execution):
    stream = [
        create_execution("first-entry", OrderSide.SELL, 1.5, 0),
        create_execution("second-entry", OrderSide.SELL, 1.1, 1),
        create_execution("first-exit", OrderSide.BUY, 1.5, 2),
        create_execution("second-exit", OrderSide.BUY, 1.1, 2),
    ]

    attributed = attribute_fill_effects(stream, _positions())

    assert {execution.execution_id: execution.effect for execution in attributed} == {
        "first-entry": "open",
        "second-entry": "increase",
        "first-exit": "reduce",
        "second-exit": "close",
    }


def test_two_entries_sharing_an_instant_open_on_the_earlier_one(create_execution):
    stream = [
        create_execution("first", OrderSide.BUY, 1.0, 0),
        create_execution("second", OrderSide.BUY, 2.0, 0),
    ]

    attributed = attribute_fill_effects(stream, _positions(_long(3.0)))

    assert {execution.execution_id: execution.effect for execution in attributed} == {
        "first": "open",
        "second": "increase",
    }


def test_each_symbol_anchored_on_its_own_position(create_execution):
    stream = [
        create_execution("btc", OrderSide.BUY, 1.0, 0),
        create_execution("eth", OrderSide.BUY, 1.0, 0, symbol=ETHUSDT),
    ]

    attributed = attribute_fill_effects(
        stream, _positions(_long(2.0), _long(1.0, ETHUSDT))
    )

    assert {execution.execution_id: execution.effect for execution in attributed} == {
        "btc": "increase",
        "eth": "open",
    }


def test_quantities_a_float_cannot_hold_exactly(create_execution):
    stream = [
        create_execution("entry-1", OrderSide.BUY, 0.1, 0),
        create_execution("entry-2", OrderSide.BUY, 0.2, 1),
        create_execution("exit", OrderSide.SELL, 0.3, 2),
    ]

    attributed = attribute_fill_effects(stream, _positions())

    assert [execution.effect for execution in attributed] == [
        "open",
        "increase",
        "close",
    ]
