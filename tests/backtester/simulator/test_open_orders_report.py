from robottraderslab.exchanges import OrderSide, StopLoss, TakeProfit

LIMIT_PRICE = 100.0
QUANTITY = 2.0
TRIGGER_PRICE = 150.0


class TestOpenOrdersReport:
    async def test_a_limit_entry_reports_its_price_quantity_and_protections(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            price=LIMIT_PRICE,
            stop_loss=StopLoss(trigger_price=90.0),
            take_profit=TakeProfit(trigger_price=120.0),
        )

        (entry,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert entry.limit_price == LIMIT_PRICE
        assert entry.quantity == QUANTITY
        assert entry.remaining_quantity == QUANTITY
        assert entry.stop_loss_price == 90.0
        assert entry.take_profit_price == 120.0

    async def test_a_limit_entry_without_protections_reports_none_for_both(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            price=LIMIT_PRICE,
        )

        (entry,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert entry.stop_loss_price is None
        assert entry.take_profit_price is None

    async def test_a_market_trigger_reports_its_levels_without_a_limit_price(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            trigger_price=TRIGGER_PRICE,
            stop_loss=StopLoss(trigger_price=90.0),
        )

        (trigger,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert trigger.limit_price is None
        assert trigger.quantity == QUANTITY
        assert trigger.remaining_quantity == QUANTITY
        assert trigger.stop_loss_price == 90.0

    async def test_a_limit_trigger_reports_the_price_it_rests_at(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            price=LIMIT_PRICE,
            trigger_price=TRIGGER_PRICE,
        )

        (trigger,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert trigger.limit_price == LIMIT_PRICE

    async def test_a_protection_on_a_position_carries_no_limit_price_or_protection(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            stop_loss=StopLoss(trigger_price=90.0),
        )
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=LIMIT_PRICE, low=95.0, high=105.0, close=LIMIT_PRICE
        )

        (stop_loss,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert stop_loss.kind == "stop-loss"
        assert stop_loss.limit_price is None
        assert stop_loss.remaining_quantity is None
        assert stop_loss.stop_loss_price is None
        assert stop_loss.take_profit_price is None
