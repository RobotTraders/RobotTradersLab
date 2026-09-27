import json

import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import OrderPlacement, OrderSide


@pytest.fixture
def placement() -> OrderPlacement:
    return OrderPlacement(
        order_id="venue-1",
        symbol=Symbol.create("BTC/USDT:USDT"),
        side=OrderSide.SELL,
        quantity=0.5,
        kind="market",
    )


def test_omits_the_reason_when_absent(placement):
    rendered = json.loads(str(placement))

    assert "reason" not in rendered


def test_includes_the_reason_when_present():
    placement = OrderPlacement(
        order_id="venue-1",
        symbol=Symbol.create("BTC/USDT:USDT"),
        side=OrderSide.SELL,
        quantity=0.5,
        kind="market",
        reason="impulse long exit",
    )

    rendered = json.loads(str(placement))

    assert rendered["reason"] == "impulse long exit"
