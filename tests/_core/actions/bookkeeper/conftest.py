from collections.abc import Callable
from datetime import datetime
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import PositionSide, PositionSnapshot
from robottraderslab.futures import FuturesAccount

PLACEMENT_RESERVE_RATE = 0.25

BTC_USDT = Symbol.create("BTC/USDT:USDT")
AUD_CAD = Symbol.create("AUD/CAD:USD")


async def _declared(exchange: Mock) -> FuturesAccount:
    account = FuturesAccount(exchange)
    await account._read_instruments([BTC_USDT, AUD_CAD])
    return account


@pytest.fixture
async def account(make_venue: Callable[..., Mock]) -> FuturesAccount:
    exchange = make_venue()
    exchange.get_quote_conversion_rate.return_value = 1.0
    exchange.placement_reserve_rate = 0.0
    exchange.placement_requirement_rate = lambda leverage: 1 / leverage
    return await _declared(exchange)


@pytest.fixture
async def reserving_account(make_venue: Callable[..., Mock]) -> FuturesAccount:
    exchange = make_venue()
    exchange.get_quote_conversion_rate.return_value = 1.0
    exchange.placement_reserve_rate = PLACEMENT_RESERVE_RATE
    exchange.placement_requirement_rate = lambda leverage: (
        1 / leverage + PLACEMENT_RESERVE_RATE
    )
    return await _declared(exchange)


@pytest.fixture
def make_position() -> Callable[[PositionSide, float], PositionSnapshot]:
    def _make_position(side: PositionSide, quantity: float) -> PositionSnapshot:
        return PositionSnapshot(
            symbol=BTC_USDT,
            side=side,
            quantity=quantity,
            average_entry_price=100.0,
            entry_time=datetime(2026, 5, 1),
            leverage=1.0,
            liquidation_price=1.0,
        )

    return _make_position
