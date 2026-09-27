import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import ExchangeRecoverableError
from robottraderslab.exchanges import OrderSide, StopLoss, TakeProfit


async def _place_limit(sim, symbol: Symbol, price: float) -> str:
    placed = await sim.exchange.place_limit_order(
        symbol=symbol, side=OrderSide.BUY, quantity=1.0, price=price
    )
    return placed.order_id


async def _open_long_with_guards(sim, symbol: Symbol) -> None:
    await sim.exchange.place_market_order(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=1.0,
        stop_loss=StopLoss(trigger_price=90.0),
        take_profit=TakeProfit(trigger_price=150.0),
    )
    sim.simulate_on_current_ohlcvs(symbol, close=100.0)


class TestCancelOrderById:
    async def test_cancels_the_matched_order(self, sim, btc_usdt_perp):
        order_id = await _place_limit(sim, btc_usdt_perp, price=90.0)

        await sim.exchange.cancel_order_by_id(btc_usdt_perp, order_id)

        assert sim.open_orders == []

    async def test_leaves_other_orders_untouched(self, sim, btc_usdt_perp):
        keeper_id = await _place_limit(sim, btc_usdt_perp, price=90.0)
        target_id = await _place_limit(sim, btc_usdt_perp, price=85.0)

        await sim.exchange.cancel_order_by_id(btc_usdt_perp, target_id)

        assert len(sim.open_orders) == 1
        assert sim.open_orders[0].order_id == keeper_id

    async def test_raises_when_order_id_not_found(self, sim, btc_usdt_perp):
        with pytest.raises(ExchangeRecoverableError, match="No open order found"):
            await sim.exchange.cancel_order_by_id(btc_usdt_perp, "nonexistent-id")

    async def test_cancels_a_position_guard_by_id(self, sim, btc_usdt_perp):
        await _open_long_with_guards(sim, btc_usdt_perp)
        stop_loss_order = next(o for o in sim.open_orders if o.kind == "stop-loss")

        await sim.exchange.cancel_order_by_id(btc_usdt_perp, stop_loss_order.order_id)

        remaining_types = {o.kind for o in sim.open_orders}
        assert "stop-loss" not in remaining_types
        assert "take-profit" in remaining_types
