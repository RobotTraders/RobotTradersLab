from robottraderslab import Symbol
from robottraderslab.exchanges import OrderSide, StopLoss, TakeProfit
from robottraderslab.strategies.futures import FuturesAccount


async def _open_long_with_guards(
    sim,
    symbol: Symbol,
    *,
    stop_loss: float = 90.0,
    take_profit: float = 150.0,
) -> None:
    await sim.exchange.place_market_order(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=1.0,
        stop_loss=StopLoss(trigger_price=stop_loss),
        take_profit=TakeProfit(trigger_price=take_profit),
    )
    sim.simulate_on_current_ohlcvs(symbol, close=100.0)


async def _place_tagged_limit(
    sim, symbol: Symbol, price: float, client_order_id: str
) -> None:
    await sim.exchange.place_limit_order(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=1.0,
        price=price,
        client_order_id=client_order_id,
    )


class TestCancelOrdersForSymbol:
    async def test_cancels_pending_limit_order(self, sim, btc_usdt_perp):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=1.0, price=90.0
        )

        await sim.exchange.cancel_orders_for_symbol(btc_usdt_perp)

        assert sim.open_orders == []

    async def test_cancels_pending_trigger_order(self, sim, btc_usdt_perp):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            price=90.0,
            trigger_price=95.0,
        )

        await sim.exchange.cancel_orders_for_symbol(btc_usdt_perp)

        assert sim.open_orders == []

    async def test_preserves_orders_for_other_symbols(
        self, sim, btc_usdt_perp, eth_usdt_perp
    ):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=1.0, price=90.0
        )
        await sim.exchange.place_limit_order(
            symbol=eth_usdt_perp, side=OrderSide.BUY, quantity=1.0, price=2000.0
        )

        await sim.exchange.cancel_orders_for_symbol(btc_usdt_perp)

        assert len(sim.open_orders) == 1
        assert sim.open_orders[0].symbol == eth_usdt_perp

    async def test_cancels_attached_position_guards(self, sim, btc_usdt_perp):
        await _open_long_with_guards(sim, btc_usdt_perp)
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=1.0, price=85.0
        )

        await sim.exchange.cancel_orders_for_symbol(btc_usdt_perp)

        assert sim.open_orders == []

    async def test_is_noop_when_no_orders_for_symbol(self, sim, btc_usdt_perp):
        await sim.exchange.cancel_orders_for_symbol(btc_usdt_perp)

        assert sim.open_orders == []


class TestCancelTaggedOrders:
    async def test_leaves_other_tags_and_protections_on_the_book(
        self, sim, btc_usdt_perp
    ):
        await _open_long_with_guards(sim, btc_usdt_perp)
        await _place_tagged_limit(sim, btc_usdt_perp, 85.0, "run-1-a")
        await _place_tagged_limit(sim, btc_usdt_perp, 80.0, "run-2-a")
        await _place_tagged_limit(sim, btc_usdt_perp, 75.0, "run-3-b")
        cancel = FuturesAccount(sim.exchange).cancel_orders(btc_usdt_perp, tag="a")

        await cancel.execute()

        remaining = {(o.kind, o.client_order_id) for o in sim.open_orders}
        assert remaining == {
            ("stop-loss", None),
            ("take-profit", None),
            ("limit", "run-3-b"),
        }

    async def test_leaves_the_same_tag_on_another_symbol(
        self, sim, btc_usdt_perp, eth_usdt_perp
    ):
        await _place_tagged_limit(sim, eth_usdt_perp, 2000.0, "run-1-a")
        cancel = FuturesAccount(sim.exchange).cancel_orders(btc_usdt_perp, tag="a")

        await cancel.execute()

        assert [o.symbol for o in sim.open_orders] == [eth_usdt_perp]
