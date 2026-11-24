import pytest
from simulation_fixture import SimulationFixture, enter_position, resting_guards

from robottraderslab import Symbol
from robottraderslab.exceptions import ExchangeRecoverableError, NoOpenPositionError
from robottraderslab.exchanges import OrderSide, TriggerType


def _get_order_by_type(sim, order_type: str):
    for order in sim.open_orders:
        if order.kind == order_type:
            return order
    return None


class TestUpdatePositionStopLoss:
    async def test_updates_stop_loss_for_long_position(self, sim, btc_usdt_perp):
        await enter_position(
            sim, btc_usdt_perp, OrderSide.BUY, stop_loss=90.0, take_profit=150.0
        )

        await sim.exchange.update_position_stop_loss(btc_usdt_perp, 85.0)

        stop_loss_order = _get_order_by_type(sim, "stop-loss")
        assert stop_loss_order.trigger_price == 85.0

    async def test_reports_the_order_it_holds(self, sim, btc_usdt_perp):
        await enter_position(
            sim, btc_usdt_perp, OrderSide.BUY, stop_loss=90.0, take_profit=150.0
        )

        placed = await sim.exchange.update_position_stop_loss(btc_usdt_perp, 85.0)

        assert placed.order_id == _get_order_by_type(sim, "stop-loss").order_id

    async def test_updates_stop_loss_for_short_position(self, sim, btc_usdt_perp):
        await enter_position(
            sim, btc_usdt_perp, OrderSide.SELL, stop_loss=110.0, take_profit=50.0
        )

        await sim.exchange.update_position_stop_loss(btc_usdt_perp, 115.0)

        stop_loss_order = _get_order_by_type(sim, "stop-loss")
        assert stop_loss_order.trigger_price == 115.0

    async def test_raises_when_no_open_position(self, sim, btc_usdt_perp):
        with pytest.raises(NoOpenPositionError):
            await sim.exchange.update_position_stop_loss(btc_usdt_perp, 85.0)

    async def test_rests_a_stop_loss_on_a_long_position_holding_none(
        self, sim, btc_usdt_perp
    ):
        await enter_position(
            sim, btc_usdt_perp, OrderSide.BUY, stop_loss=None, take_profit=150.0
        )

        placed = await sim.exchange.update_position_stop_loss(btc_usdt_perp, 85.0)

        stop_loss_order = _get_order_by_type(sim, "stop-loss")
        assert stop_loss_order.order_id == placed.order_id
        assert stop_loss_order.trigger_price == 85.0
        assert stop_loss_order.side == OrderSide.SELL
        assert stop_loss_order.quantity is None

    async def test_rests_a_stop_loss_on_a_short_position_holding_none(
        self, sim, btc_usdt_perp
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.SELL)

        await sim.exchange.update_position_stop_loss(btc_usdt_perp, 115.0)

        stop_loss_order = _get_order_by_type(sim, "stop-loss")
        assert stop_loss_order.trigger_price == 115.0
        assert stop_loss_order.side == OrderSide.BUY

    @pytest.mark.parametrize(
        "trigger_type", [TriggerType.MARK_PRICE, TriggerType.LAST_PRICE]
    )
    async def test_the_stop_loss_it_rests_watches_the_trigger_type_given(
        self, sim, btc_usdt_perp, trigger_type
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY)

        await sim.exchange.update_position_stop_loss(btc_usdt_perp, 85.0, trigger_type)

        assert _get_order_by_type(sim, "stop-loss").trigger_type == trigger_type

    async def test_the_stop_loss_it_rests_fires_when_price_reaches_it(
        self, sim, btc_usdt_perp
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY)
        await sim.exchange.update_position_stop_loss(btc_usdt_perp, 95.0)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=96.0, low=94.0, high=100.0)

        assert sim.open_positions == {}


class TestUpdateWithGuardsOfSeveralEntries:
    async def _open_long_then_add_with_stop_losses(
        self, sim: SimulationFixture, symbol: Symbol
    ) -> None:
        await enter_position(
            sim, symbol, OrderSide.BUY, stop_loss=90.0, take_profit=110.0
        )
        await enter_position(sim, symbol, OrderSide.BUY, stop_loss=95.0)

    async def test_a_stop_loss_update_leaves_one_whole_position_stop_loss(
        self, sim, btc_usdt_perp
    ):
        await self._open_long_then_add_with_stop_losses(sim, btc_usdt_perp)

        await sim.exchange.update_position_stop_loss(btc_usdt_perp, 85.0)

        assert resting_guards(sim, "stop-loss") == [(85.0, None)]

    async def test_a_take_profit_update_leaves_one_whole_position_take_profit(
        self, sim, btc_usdt_perp
    ):
        await enter_position(
            sim, btc_usdt_perp, OrderSide.BUY, stop_loss=90.0, take_profit=120.0
        )
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY, take_profit=110.0)

        await sim.exchange.update_position_take_profit(btc_usdt_perp, 130.0)

        assert resting_guards(sim, "take-profit") == [(130.0, None)]

    async def test_the_updated_stop_loss_survives_a_sized_take_profit_firing(
        self, sim, btc_usdt_perp
    ):
        await self._open_long_then_add_with_stop_losses(sim, btc_usdt_perp)
        await sim.exchange.update_position_stop_loss(btc_usdt_perp, 85.0)

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=100.0, low=100.0, high=111.0, close=105.0
        )

        assert sim.open_positions[btc_usdt_perp].quantity == 1.0
        assert [order.trigger_price for order in sim.open_orders] == [85.0]


class TestPositionProtectionLevels:
    async def test_a_read_without_protections_reports_no_level(
        self, sim, btc_usdt_perp
    ):
        await enter_position(
            sim, btc_usdt_perp, OrderSide.BUY, stop_loss=90.0, take_profit=150.0
        )

        positions = await sim.exchange.get_open_positions([btc_usdt_perp])

        assert positions[btc_usdt_perp].stop_loss_price is None
        assert positions[btc_usdt_perp].take_profit_price is None

    async def test_a_read_with_protections_reports_the_booked_levels(
        self, sim, btc_usdt_perp
    ):
        await enter_position(
            sim, btc_usdt_perp, OrderSide.BUY, stop_loss=90.0, take_profit=150.0
        )

        positions = await sim.exchange.get_open_positions(
            [btc_usdt_perp], with_protections=True
        )

        assert positions[btc_usdt_perp].stop_loss_price == 90.0
        assert positions[btc_usdt_perp].take_profit_price == 150.0

    async def test_moved_protections_read_back_on_the_position(
        self, sim, btc_usdt_perp
    ):
        await enter_position(
            sim, btc_usdt_perp, OrderSide.BUY, stop_loss=90.0, take_profit=150.0
        )
        await sim.exchange.update_position_stop_loss(btc_usdt_perp, 85.0)
        await sim.exchange.update_position_take_profit(btc_usdt_perp, 160.0)

        positions = await sim.exchange.get_open_positions(
            [btc_usdt_perp], with_protections=True
        )

        assert positions[btc_usdt_perp].stop_loss_price == 85.0
        assert positions[btc_usdt_perp].take_profit_price == 160.0

    async def test_a_position_opened_bare(self, sim, btc_usdt_perp):
        await enter_position(
            sim, btc_usdt_perp, OrderSide.BUY, stop_loss=None, take_profit=None
        )

        positions = await sim.exchange.get_open_positions(
            [btc_usdt_perp], with_protections=True
        )

        assert positions[btc_usdt_perp].stop_loss_price is None
        assert positions[btc_usdt_perp].take_profit_price is None


class TestUpdatePositionTakeProfit:
    async def test_updates_take_profit_for_long_position(self, sim, btc_usdt_perp):
        await enter_position(
            sim, btc_usdt_perp, OrderSide.BUY, stop_loss=90.0, take_profit=150.0
        )

        await sim.exchange.update_position_take_profit(btc_usdt_perp, 160.0)

        take_profit_order = _get_order_by_type(sim, "take-profit")
        assert take_profit_order.trigger_price == 160.0

    async def test_reports_the_order_it_holds(self, sim, btc_usdt_perp):
        await enter_position(
            sim, btc_usdt_perp, OrderSide.BUY, stop_loss=90.0, take_profit=150.0
        )

        placed = await sim.exchange.update_position_take_profit(btc_usdt_perp, 140.0)

        assert placed.order_id == _get_order_by_type(sim, "take-profit").order_id

    async def test_updates_take_profit_for_short_position(self, sim, btc_usdt_perp):
        await enter_position(
            sim, btc_usdt_perp, OrderSide.SELL, stop_loss=110.0, take_profit=50.0
        )

        await sim.exchange.update_position_take_profit(btc_usdt_perp, 40.0)

        take_profit_order = _get_order_by_type(sim, "take-profit")
        assert take_profit_order.trigger_price == 40.0

    async def test_raises_when_no_open_position(self, sim, btc_usdt_perp):
        with pytest.raises(NoOpenPositionError):
            await sim.exchange.update_position_take_profit(btc_usdt_perp, 160.0)

    async def test_raises_when_position_has_no_take_profit(self, sim, btc_usdt_perp):
        await enter_position(
            sim, btc_usdt_perp, OrderSide.BUY, stop_loss=90.0, take_profit=None
        )

        with pytest.raises(ExchangeRecoverableError, match="No active take-profit"):
            await sim.exchange.update_position_take_profit(btc_usdt_perp, 160.0)
