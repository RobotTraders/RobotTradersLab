from robottraderslab.exchanges import OrderSide, StopLoss, TakeProfit


class TestGuardOrdersOwnedByPosition:
    async def test_a_reduce_only_close_cancels_the_resting_guards(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            stop_loss=StopLoss(trigger_price=90.0),
            take_profit=TakeProfit(trigger_price=150.0),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)
        assert len(sim.open_orders) == 2

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=1.0, reduce_only=True
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=101.0)

        assert sim.open_orders == []
        assert len(sim.open_positions) == 0

    async def test_pending_entry_order_survives_reduce_only_close(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            stop_loss=StopLoss(trigger_price=90.0),
            take_profit=TakeProfit(trigger_price=150.0),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        placed = await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=1.0, price=85.0
        )

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=1.0, reduce_only=True
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        assert len(sim.open_orders) == 1
        assert sim.open_orders[0].order_id == placed.order_id
        assert sim.open_orders[0].kind == "limit"

    async def test_partial_close_leaves_guards_resting_for_remaining_quantity(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            stop_loss=StopLoss(trigger_price=90.0),
            take_profit=TakeProfit(trigger_price=150.0),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)
        assert len(sim.open_orders) == 2

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=1.0, reduce_only=True
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        assert len(sim.open_orders) == 2
        assert sim.open_positions[btc_usdt_perp].quantity == 1.0

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=80.0, high=100.0, close=85.0)

        assert sim.open_orders == []
        assert len(sim.open_positions) == 0
        fills = sim.fill_recorder.get_fills()
        exit_fills = fills[fills["fill_type"] == "exit_long"]
        assert exit_fills.iloc[-1]["gross_quantity"] == 1.0

    async def test_guards_from_separate_orders_on_same_position_are_both_cancelled(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            stop_loss=StopLoss(trigger_price=90.0),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            take_profit=TakeProfit(trigger_price=150.0),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)
        assert len(sim.open_orders) == 2

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=2.0, reduce_only=True
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        assert sim.open_orders == []
        assert len(sim.open_positions) == 0

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=101.0)

        assert sim.open_orders == []

    async def test_full_close_preserves_guards_on_other_symbols(
        self, sim, btc_usdt_perp, eth_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            stop_loss=StopLoss(trigger_price=90.0),
            take_profit=TakeProfit(trigger_price=150.0),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        await sim.exchange.place_market_order(
            symbol=eth_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            stop_loss=StopLoss(trigger_price=1_800.0),
            take_profit=TakeProfit(trigger_price=2_200.0),
        )
        sim.simulate_on_current_ohlcvs(eth_usdt_perp, close=2_000.0)

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=1.0, reduce_only=True
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        assert btc_usdt_perp not in sim.open_positions
        assert eth_usdt_perp in sim.open_positions
        assert len(sim.open_orders) == 2
        assert all(order.symbol == eth_usdt_perp for order in sim.open_orders)

    async def test_opposing_entry_full_close_preserves_unrelated_pending_order(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            stop_loss=StopLoss(trigger_price=90.0),
            take_profit=TakeProfit(trigger_price=150.0),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        placed = await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=1.0, price=85.0
        )

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=1.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        assert len(sim.open_positions) == 0
        assert len(sim.open_orders) == 1
        assert sim.open_orders[0].order_id == placed.order_id

    async def test_liquidation_preserves_unrelated_pending_order(
        self, sim, btc_usdt_perp
    ):
        sim.simulation_engine.set_symbol_leverage(btc_usdt_perp, 5.0)
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=0.5
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=50_000.0)

        placed = await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=0.1, price=35_000.0
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, low=39_000.0, high=42_000.0, close=41_000.0
        )

        assert len(sim.open_positions) == 0
        assert len(sim.open_orders) == 1
        assert sim.open_orders[0].order_id == placed.order_id

    async def test_guard_fires_alongside_pending_close_order_in_same_tick(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            stop_loss=StopLoss(trigger_price=90.0),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=1.0,
            price=95.0,
            reduce_only=True,
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=80.0, high=100.0, close=85.0)

        assert sim.open_orders == []
        assert len(sim.open_positions) == 0

    async def test_pending_trigger_entry_survives_a_stop_loss_close(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            stop_loss=StopLoss(trigger_price=90.0),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            price=120.0,
            trigger_price=130.0,
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=85.0, high=85.0, close=85.0)

        assert len(sim.open_positions) == 0
        assert [order.kind for order in sim.open_orders] == ["trigger"]

    async def test_a_flip_arms_the_new_position_with_the_orders_guards(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=1.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=3.0,
            stop_loss=StopLoss(trigger_price=90.0),
            take_profit=TakeProfit(trigger_price=150.0),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        assert sim.open_positions[btc_usdt_perp].quantity == 2.0
        assert sorted(order.kind for order in sim.open_orders) == [
            "stop-loss",
            "take-profit",
        ]

    async def test_a_flip_closing_the_guards_of_the_side_it_replaces(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=1.0,
            stop_loss=StopLoss(trigger_price=110.0),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=3.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        assert sim.open_positions[btc_usdt_perp].side.value == "long"
        assert sim.open_orders == []

    async def test_a_liquidation_that_no_guard_took_part_in(self, sim, btc_usdt_perp):
        sim.simulation_engine.set_symbol_leverage(btc_usdt_perp, 5.0)
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            take_profit=TakeProfit(trigger_price=100_000.0),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=50_000.0)
        assert len(sim.open_orders) == 1

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, low=39_000.0, high=41_000.0, close=40_000.0
        )

        assert len(sim.open_positions) == 0
        assert sim.open_orders == []
