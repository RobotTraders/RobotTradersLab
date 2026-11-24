import pytest

from robottraderslab.exchanges import (
    OrderSide,
    StopLoss,
    TakeProfit,
    TimeInForce,
    TriggerType,
)

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

    async def test_a_trigger_entry_is_listed_whatever_the_triggers_keyword(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            trigger_price=TRIGGER_PRICE,
        )

        listed = await sim.exchange.get_open_orders([btc_usdt_perp])
        listed_with_triggers = await sim.exchange.get_open_orders(
            [btc_usdt_perp], with_triggers=True
        )

        assert len(listed) == 1
        assert listed_with_triggers == listed

    async def test_a_limit_entry_states_no_trailing_distance(self, sim, btc_usdt_perp):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            price=LIMIT_PRICE,
        )

        (entry,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert entry.trailing_distance is None

    async def test_a_trigger_entry_states_no_trailing_distance(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            trigger_price=TRIGGER_PRICE,
        )

        (trigger,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert trigger.trailing_distance is None

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

    async def test_a_protection_on_a_position_reports_the_quantity_of_its_entry_and_no_limit_price_or_protection(
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
        assert stop_loss.remaining_quantity == QUANTITY
        assert stop_loss.stop_loss_price is None
        assert stop_loss.take_profit_price is None
        assert stop_loss.trailing_distance is None


class TestReduceOnlyReport:
    async def test_a_reduce_only_exit_reports_true(self, sim, btc_usdt_perp):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=QUANTITY,
            price=LIMIT_PRICE,
            reduce_only=True,
        )

        (exit_order,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert exit_order.reduce_only is True

    async def test_an_entry_placed_without_reduce_only(self, sim, btc_usdt_perp):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            price=LIMIT_PRICE,
        )

        (entry,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert entry.reduce_only is False

    async def test_a_reduce_only_trigger_reports_the_flag_of_its_order(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=QUANTITY,
            reduce_only=True,
            trigger_price=TRIGGER_PRICE,
        )

        (trigger,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert trigger.reduce_only is True

    async def test_a_stop_loss_and_a_take_profit(self, sim, btc_usdt_perp):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            stop_loss=StopLoss(trigger_price=90.0),
            take_profit=TakeProfit(trigger_price=120.0),
        )
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=LIMIT_PRICE, low=95.0, high=105.0, close=LIMIT_PRICE
        )

        protections = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert [order.kind for order in protections] == ["stop-loss", "take-profit"]
        assert [order.reduce_only for order in protections] == [True, True]


class TestTimeInForceReport:
    async def test_a_post_only_limit_reports_post_only(self, sim, btc_usdt_perp):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            price=LIMIT_PRICE,
            time_in_force=TimeInForce.POST_ONLY,
        )

        (entry,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert entry.time_in_force is TimeInForce.POST_ONLY

    async def test_a_limit_trigger_reports_the_time_in_force_of_its_order(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            price=LIMIT_PRICE,
            trigger_price=TRIGGER_PRICE,
            time_in_force=TimeInForce.POST_ONLY,
        )

        (trigger,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert trigger.time_in_force is TimeInForce.POST_ONLY

    async def test_a_market_trigger(self, sim, btc_usdt_perp):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            trigger_price=TRIGGER_PRICE,
        )

        (trigger,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert trigger.time_in_force is None

    async def test_a_protection(self, sim, btc_usdt_perp):
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

        assert stop_loss.time_in_force is None


class TestTriggerTypeReport:
    @pytest.mark.parametrize(
        "trigger_type", [TriggerType.MARK_PRICE, TriggerType.LAST_PRICE]
    )
    async def test_a_trigger_reports_the_price_it_was_placed_to_watch(
        self, sim, btc_usdt_perp, trigger_type
    ):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            price=LIMIT_PRICE,
            trigger_price=TRIGGER_PRICE,
            trigger_type=trigger_type,
        )

        (trigger,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert trigger.trigger_type is trigger_type

    async def test_a_market_trigger_reports_the_price_it_was_placed_to_watch(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            trigger_price=TRIGGER_PRICE,
            trigger_type=TriggerType.INDEX_PRICE,
        )

        (trigger,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert trigger.trigger_type is TriggerType.INDEX_PRICE

    async def test_an_order_without_a_trigger(self, sim, btc_usdt_perp):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            price=LIMIT_PRICE,
            trigger_type=TriggerType.MARK_PRICE,
        )

        (entry,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert entry.trigger_type is None

    async def test_an_entry_reports_the_types_of_its_attached_protections(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            price=LIMIT_PRICE,
            stop_loss=StopLoss(trigger_price=90.0, trigger_type=TriggerType.MARK_PRICE),
            take_profit=TakeProfit(
                trigger_price=120.0, trigger_type=TriggerType.LAST_PRICE
            ),
        )

        (entry,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert entry.stop_loss_trigger_type is TriggerType.MARK_PRICE
        assert entry.take_profit_trigger_type is TriggerType.LAST_PRICE

    async def test_a_trigger_reports_the_types_of_its_attached_protections(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            trigger_price=TRIGGER_PRICE,
            stop_loss=StopLoss(
                trigger_price=90.0, trigger_type=TriggerType.INDEX_PRICE
            ),
        )

        (trigger,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert trigger.stop_loss_trigger_type is TriggerType.INDEX_PRICE
        assert trigger.take_profit_trigger_type is None

    async def test_an_entry_without_protections(self, sim, btc_usdt_perp):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            price=LIMIT_PRICE,
        )

        (entry,) = await sim.exchange.get_open_orders([btc_usdt_perp])

        assert entry.stop_loss_trigger_type is None
        assert entry.take_profit_trigger_type is None

    @staticmethod
    async def _protections_on_a_position(sim, symbol):
        await sim.exchange.place_market_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            stop_loss=StopLoss(trigger_price=90.0, trigger_type=TriggerType.MARK_PRICE),
            take_profit=TakeProfit(
                trigger_price=120.0, trigger_type=TriggerType.LAST_PRICE
            ),
        )
        sim.simulate_on_current_ohlcvs(
            symbol, open=LIMIT_PRICE, low=95.0, high=105.0, close=LIMIT_PRICE
        )
        return await sim.exchange.get_open_orders([symbol])

    async def test_a_protection_on_a_position_reports_the_type_it_was_booked_with(
        self, sim, btc_usdt_perp
    ):
        stop_loss, take_profit = await self._protections_on_a_position(
            sim, btc_usdt_perp
        )

        assert stop_loss.trigger_type is TriggerType.MARK_PRICE
        assert take_profit.trigger_type is TriggerType.LAST_PRICE

    async def test_a_protection_on_a_position_carries_no_attached_types(
        self, sim, btc_usdt_perp
    ):
        stop_loss, take_profit = await self._protections_on_a_position(
            sim, btc_usdt_perp
        )

        assert stop_loss.stop_loss_trigger_type is None
        assert take_profit.take_profit_trigger_type is None

    async def test_a_moved_protection_reports_the_type_it_was_moved_with(
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

        await sim.exchange.update_position_stop_loss(
            btc_usdt_perp, 92.0, TriggerType.INDEX_PRICE
        )

        (stop_loss,) = await sim.exchange.get_open_orders([btc_usdt_perp])
        assert stop_loss.trigger_type is TriggerType.INDEX_PRICE

    async def test_a_moved_take_profit_reports_the_type_it_was_moved_with(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            take_profit=TakeProfit(trigger_price=120.0),
        )
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=LIMIT_PRICE, low=95.0, high=105.0, close=LIMIT_PRICE
        )

        await sim.exchange.update_position_take_profit(
            btc_usdt_perp, 118.0, TriggerType.INDEX_PRICE
        )

        (take_profit,) = await sim.exchange.get_open_orders([btc_usdt_perp])
        assert take_profit.trigger_type is TriggerType.INDEX_PRICE
