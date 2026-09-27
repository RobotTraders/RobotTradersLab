from robottraderslab import Symbol
from robottraderslab.exchanges import Execution, OrderSide, StopLoss, TakeProfit


class TestFillReasons:
    entry_close = 100.0
    exit_close = 110.0
    stop_loss = 90.0
    take_profit = 120.0
    below_stop_loss = 85.0
    above_take_profit = 125.0
    entry_trigger = 150.0
    above_entry_trigger = 155.0

    async def enter_long(self, sim, btc_usdt_perp: Symbol, reason: str | None = None):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=1.0, reason=reason
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.entry_close)

    async def sell(self, sim, btc_usdt_perp: Symbol, quantity: float):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.exit_close)
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=quantity
        )

    async def enter_long_guarded(
        self, sim, btc_usdt_perp: Symbol, stop_loss_reason: str | None = None
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            reason="entry",
            stop_loss=StopLoss(trigger_price=self.stop_loss, reason=stop_loss_reason),
            take_profit=TakeProfit(trigger_price=self.take_profit),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.entry_close)

    async def test_describer_names_a_fill_whose_order_states_no_reason(
        self, sim, btc_usdt_perp
    ):
        sim.simulation_engine.describe_fills_with(lambda fill: f"described {fill.kind}")

        await self.enter_long(sim, btc_usdt_perp)

        assert list(sim.fill_recorder.get_fills()["reason"]) == ["described market"]

    async def test_an_orders_own_reason_wins_over_the_describer(
        self, sim, btc_usdt_perp
    ):
        sim.simulation_engine.describe_fills_with(lambda fill: "described")

        await self.enter_long(sim, btc_usdt_perp, reason="breakout")

        assert list(sim.fill_recorder.get_fills()["reason"]) == ["breakout"]

    async def test_no_describer_leaves_a_reasonless_fill_unnamed(
        self, sim, btc_usdt_perp
    ):
        await self.enter_long(sim, btc_usdt_perp)

        assert sim.fill_recorder.get_fills()["reason"].isna().all()

    async def test_describer_names_both_legs_of_a_flip(self, sim, btc_usdt_perp):
        sim.simulation_engine.describe_fills_with(
            lambda fill: f"described {fill.side.value}"
        )

        await self.enter_long(sim, btc_usdt_perp)
        await self.sell(sim, btc_usdt_perp, quantity=2.0)

        fills = sim.fill_recorder.get_fills()
        assert list(fills["fill_type"]) == ["enter_long", "exit_long", "enter_short"]
        assert list(fills["reason"]) == [
            "described buy",
            "described sell",
            "described sell",
        ]

    async def test_describer_sees_the_profit_the_exit_realised(
        self, sim, btc_usdt_perp
    ):
        def describe(fill: Execution) -> str:
            return f"realised {fill.realised_profit:+.0f} on {fill.quantity:.0f}"

        sim.simulation_engine.describe_fills_with(describe)

        await self.enter_long(sim, btc_usdt_perp)
        await self.sell(sim, btc_usdt_perp, quantity=1.0)

        assert list(sim.fill_recorder.get_fills()["reason"]) == [
            "realised +0 on 1",
            "realised +10 on 1",
        ]

    async def test_an_unnamed_stop_loss_is_reported_by_its_kind(
        self, sim, btc_usdt_perp
    ):
        await self.enter_long_guarded(sim, btc_usdt_perp)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.below_stop_loss)

        assert list(sim.fill_recorder.get_fills()["reason"]) == ["entry", "stop-loss"]

    async def test_an_unnamed_take_profit_is_reported_by_its_kind(
        self, sim, btc_usdt_perp
    ):
        await self.enter_long_guarded(sim, btc_usdt_perp)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.above_take_profit)

        assert list(sim.fill_recorder.get_fills()["reason"]) == ["entry", "take-profit"]

    async def test_a_stop_loss_the_describer_leaves_unnamed_falls_back_to_its_kind(
        self, sim, btc_usdt_perp
    ):
        sim.simulation_engine.describe_fills_with(lambda fill: None)

        await self.enter_long_guarded(sim, btc_usdt_perp)
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.below_stop_loss)

        assert list(sim.fill_recorder.get_fills()["reason"]) == ["entry", "stop-loss"]

    async def test_a_describers_name_wins_over_the_protection_kind(
        self, sim, btc_usdt_perp
    ):
        sim.simulation_engine.describe_fills_with(lambda fill: "band stop hit")

        await self.enter_long_guarded(sim, btc_usdt_perp)
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.below_stop_loss)

        assert list(sim.fill_recorder.get_fills()["reason"]) == [
            "entry",
            "band stop hit",
        ]

    async def test_the_reason_a_stop_loss_carries_wins_over_its_kind(
        self, sim, btc_usdt_perp
    ):
        await self.enter_long_guarded(sim, btc_usdt_perp, stop_loss_reason="hard stop")

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.below_stop_loss)

        assert list(sim.fill_recorder.get_fills()["reason"]) == ["entry", "hard stop"]

    async def test_an_unnamed_stop_loss_is_named_when_nothing_records_executions(
        self, sim_without_recording, btc_usdt_perp
    ):
        await self.enter_long_guarded(sim_without_recording, btc_usdt_perp)

        sim_without_recording.simulate_on_current_ohlcvs(
            btc_usdt_perp, close=self.below_stop_loss
        )

        fills = sim_without_recording.fill_recorder.get_fills()
        assert list(fills["reason"]) == ["entry", "stop-loss"]

    async def test_a_fired_entry_trigger_nobody_named_stays_disowned(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            trigger_price=self.entry_trigger,
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.above_entry_trigger)

        assert sim.fill_recorder.get_fills()["reason"].isna().all()
