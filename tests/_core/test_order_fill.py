import json
from datetime import datetime, timezone

import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import OrderFill, OrderSide


@pytest.fixture
def btc_symbol() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def order_fill(btc_symbol) -> OrderFill:
    return OrderFill(
        order_id="123456",
        symbol=btc_symbol,
        side=OrderSide.BUY,
        quantity=0.5,
        kind="market",
        timestamp=datetime(2025, 11, 24, 10, 30, 0, tzinfo=timezone.utc),
    )


class TestOrderFillStr:
    def test_returns_valid_json(self, order_fill):
        result = str(order_fill)

        parsed = json.loads(result)

        assert isinstance(parsed, dict)

    def test_contains_all_fields(self, order_fill):
        result = json.loads(str(order_fill))

        assert "time" in result
        assert "symbol" in result
        assert "side" in result
        assert "kind" in result
        assert "qty" in result
        assert "order_id" in result

    def test_field_values(self, order_fill, btc_symbol):
        result = json.loads(str(order_fill))

        assert result["symbol"] == str(btc_symbol)
        assert result["side"] == "buy"
        assert result["kind"] == "market"
        assert result["qty"] == 0.5
        assert result["order_id"] == "123456"

    def test_time_format(self, order_fill):
        result = json.loads(str(order_fill))

        assert "2025-11-24" in result["time"]
        assert "10:30:00" in result["time"]

    def test_omits_value_profit_and_reason_when_absent(self, order_fill):
        result = json.loads(str(order_fill))

        assert "value" not in result
        assert "profit" not in result
        assert "reason" not in result

    def test_omits_effect_and_source_when_absent(self, order_fill):
        parsed = json.loads(str(order_fill))

        assert "effect" not in parsed
        assert "source" not in parsed

    def test_includes_effect_and_source_when_present(self, btc_symbol):
        order_fill = OrderFill(
            order_id="123456",
            symbol=btc_symbol,
            side=OrderSide.SELL,
            quantity=0.5,
            kind="take-profit",
            effect="close",
            source="take-profit",
        )

        parsed = json.loads(str(order_fill))

        assert parsed["effect"] == "close"
        assert parsed["source"] == "take-profit"

    def test_includes_value_profit_and_reason_when_present(self, btc_symbol):
        order_fill = OrderFill(
            order_id="123456",
            symbol=btc_symbol,
            side=OrderSide.BUY,
            quantity=0.5,
            kind="market",
            filled_value=366.294,
            realised_profit=-12.5,
            reason="entry rung 2 of 4",
        )

        result = json.loads(str(order_fill))

        assert result["value"] == 366.294
        assert result["profit"] == -12.5
        assert result["reason"] == "entry rung 2 of 4"
