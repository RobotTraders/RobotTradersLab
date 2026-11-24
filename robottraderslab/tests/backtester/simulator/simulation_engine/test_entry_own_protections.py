from gapped_entry_helpers import place_trigger_entry
from simulation_fixture import enter_position, resting_guards

from robottraderslab.exchanges import OrderSide

ENTRY_PRICE = 100.0


class TestEntryOwnProtections:
    async def test_an_add_stop_loss_closes_only_the_add(self, sim, btc_usdt_perp):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY, stop_loss=90.0)
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY, stop_loss=95.0)

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=ENTRY_PRICE, low=94.0, high=ENTRY_PRICE, close=96.0
        )

        fills = sim.fill_recorder.get_fills()
        assert list(fills["fill_type"])[-1] == "reduce_long"
        assert fills["price"].iloc[-1] == 95.0
        assert fills["gross_quantity"].iloc[-1] == 1.0
        assert sim.open_positions[btc_usdt_perp].quantity == 1.0
        assert resting_guards(sim, "stop-loss") == [(90.0, 1.0)]

    async def test_an_add_take_profit_closes_only_the_add(self, sim, btc_usdt_perp):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY, take_profit=120.0)
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY, take_profit=110.0)

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=ENTRY_PRICE, low=ENTRY_PRICE, high=111.0, close=105.0
        )

        fills = sim.fill_recorder.get_fills()
        assert list(fills["fill_type"])[-1] == "reduce_long"
        assert fills["price"].iloc[-1] == 110.0
        assert fills["gross_quantity"].iloc[-1] == 1.0
        assert sim.open_positions[btc_usdt_perp].quantity == 1.0
        assert resting_guards(sim, "take-profit") == [(120.0, 1.0)]

    async def test_a_guard_firing_cancels_only_the_sibling_of_its_entry(
        self, sim, btc_usdt_perp
    ):
        await enter_position(
            sim, btc_usdt_perp, OrderSide.BUY, stop_loss=90.0, take_profit=120.0
        )
        await enter_position(
            sim, btc_usdt_perp, OrderSide.BUY, stop_loss=95.0, take_profit=110.0
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=ENTRY_PRICE, low=94.0, high=ENTRY_PRICE, close=96.0
        )

        assert resting_guards(sim, "stop-loss") == [(90.0, 1.0)]
        assert resting_guards(sim, "take-profit") == [(120.0, 1.0)]

    async def test_an_add_filled_past_its_own_take_profit_closes_only_the_add(
        self, sim, btc_usdt_perp
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY, take_profit=150.0)
        await place_trigger_entry(
            sim,
            btc_usdt_perp,
            OrderSide.BUY,
            placement_close=95.0,
            take_profit=102.0,
            quantity=1.0,
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=103.0, low=102.5, high=104.0, close=103.0
        )

        fills = sim.fill_recorder.get_fills()
        assert list(fills["fill_type"])[-2:] == ["add_to_long", "reduce_long"]
        assert list(fills["price"])[-2:] == [103.0, 103.0]
        assert fills["gross_quantity"].iloc[-1] == 1.0
        assert sim.open_positions[btc_usdt_perp].quantity == 1.0
        assert resting_guards(sim, "take-profit") == [(150.0, 1.0)]
