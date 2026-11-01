import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import ExchangeRecoverableError, NoOpenPositionError
from robottraderslab.exchanges import OrderSide, StopLoss, TakeProfit


async def _open_long_with_tp_sl(
    sim,
    symbol: Symbol,
    *,
    stop_loss: float | None = 90.0,
    take_profit: float | None = 150.0,
) -> None:
    await sim.exchange.place_market_order(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=1.0,
        stop_loss=StopLoss(trigger_price=stop_loss) if stop_loss is not None else None,
        take_profit=TakeProfit(trigger_price=take_profit)
        if take_profit is not None
        else None,
    )
    sim.simulate_on_current_ohlcvs(symbol, close=100.0)


async def _open_short_with_tp_sl(
    sim,
    symbol: Symbol,
    *,
    stop_loss: float | None = 110.0,
    take_profit: float | None = 50.0,
) -> None:
    await sim.exchange.place_market_order(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=1.0,
        stop_loss=StopLoss(trigger_price=stop_loss) if stop_loss is not None else None,
        take_profit=TakeProfit(trigger_price=take_profit)
        if take_profit is not None
        else None,
    )
    sim.simulate_on_current_ohlcvs(symbol, close=100.0)


def _get_order_by_type(sim, order_type: str):
    for order in sim.open_orders:
        if order.kind == order_type:
            return order
    return None


class TestUpdatePositionStopLoss:
    async def test_updates_stop_loss_for_long_position(self, sim, btc_usdt_perp):
        await _open_long_with_tp_sl(sim, btc_usdt_perp)

        await sim.exchange.update_position_stop_loss(btc_usdt_perp, 85.0)

        stop_loss_order = _get_order_by_type(sim, "stop-loss")
        assert stop_loss_order.trigger_price == 85.0

    async def test_reports_the_order_it_holds(self, sim, btc_usdt_perp):
        await _open_long_with_tp_sl(sim, btc_usdt_perp)

        placed = await sim.exchange.update_position_stop_loss(btc_usdt_perp, 85.0)

        assert placed.order_id == _get_order_by_type(sim, "stop-loss").order_id

    async def test_updates_stop_loss_for_short_position(self, sim, btc_usdt_perp):
        await _open_short_with_tp_sl(sim, btc_usdt_perp)

        await sim.exchange.update_position_stop_loss(btc_usdt_perp, 115.0)

        stop_loss_order = _get_order_by_type(sim, "stop-loss")
        assert stop_loss_order.trigger_price == 115.0

    async def test_raises_when_no_open_position(self, sim, btc_usdt_perp):
        with pytest.raises(NoOpenPositionError):
            await sim.exchange.update_position_stop_loss(btc_usdt_perp, 85.0)

    async def test_raises_when_position_has_no_stop_loss(self, sim, btc_usdt_perp):
        await _open_long_with_tp_sl(
            sim, btc_usdt_perp, stop_loss=None, take_profit=150.0
        )

        with pytest.raises(ExchangeRecoverableError, match="No active stop-loss"):
            await sim.exchange.update_position_stop_loss(btc_usdt_perp, 85.0)


class TestPositionProtectionLevels:
    async def test_a_read_without_protections_reports_no_level(
        self, sim, btc_usdt_perp
    ):
        await _open_long_with_tp_sl(sim, btc_usdt_perp)

        positions = await sim.exchange.get_open_positions([btc_usdt_perp])

        assert positions[btc_usdt_perp].stop_loss_price is None
        assert positions[btc_usdt_perp].take_profit_price is None

    async def test_a_read_with_protections_reports_the_booked_levels(
        self, sim, btc_usdt_perp
    ):
        await _open_long_with_tp_sl(sim, btc_usdt_perp)

        positions = await sim.exchange.get_open_positions(
            [btc_usdt_perp], with_protections=True
        )

        assert positions[btc_usdt_perp].stop_loss_price == 90.0
        assert positions[btc_usdt_perp].take_profit_price == 150.0

    async def test_moved_protections_read_back_on_the_position(
        self, sim, btc_usdt_perp
    ):
        await _open_long_with_tp_sl(sim, btc_usdt_perp)
        await sim.exchange.update_position_stop_loss(btc_usdt_perp, 85.0)
        await sim.exchange.update_position_take_profit(btc_usdt_perp, 160.0)

        positions = await sim.exchange.get_open_positions(
            [btc_usdt_perp], with_protections=True
        )

        assert positions[btc_usdt_perp].stop_loss_price == 85.0
        assert positions[btc_usdt_perp].take_profit_price == 160.0

    async def test_a_position_opened_bare(self, sim, btc_usdt_perp):
        await _open_long_with_tp_sl(
            sim, btc_usdt_perp, stop_loss=None, take_profit=None
        )

        positions = await sim.exchange.get_open_positions(
            [btc_usdt_perp], with_protections=True
        )

        assert positions[btc_usdt_perp].stop_loss_price is None
        assert positions[btc_usdt_perp].take_profit_price is None


class TestUpdatePositionTakeProfit:
    async def test_updates_take_profit_for_long_position(self, sim, btc_usdt_perp):
        await _open_long_with_tp_sl(sim, btc_usdt_perp)

        await sim.exchange.update_position_take_profit(btc_usdt_perp, 160.0)

        take_profit_order = _get_order_by_type(sim, "take-profit")
        assert take_profit_order.trigger_price == 160.0

    async def test_reports_the_order_it_holds(self, sim, btc_usdt_perp):
        await _open_long_with_tp_sl(sim, btc_usdt_perp)

        placed = await sim.exchange.update_position_take_profit(btc_usdt_perp, 140.0)

        assert placed.order_id == _get_order_by_type(sim, "take-profit").order_id

    async def test_updates_take_profit_for_short_position(self, sim, btc_usdt_perp):
        await _open_short_with_tp_sl(sim, btc_usdt_perp)

        await sim.exchange.update_position_take_profit(btc_usdt_perp, 40.0)

        take_profit_order = _get_order_by_type(sim, "take-profit")
        assert take_profit_order.trigger_price == 40.0

    async def test_raises_when_no_open_position(self, sim, btc_usdt_perp):
        with pytest.raises(NoOpenPositionError):
            await sim.exchange.update_position_take_profit(btc_usdt_perp, 160.0)

    async def test_raises_when_position_has_no_take_profit(self, sim, btc_usdt_perp):
        await _open_long_with_tp_sl(
            sim, btc_usdt_perp, stop_loss=90.0, take_profit=None
        )

        with pytest.raises(ExchangeRecoverableError, match="No active take-profit"):
            await sim.exchange.update_position_take_profit(btc_usdt_perp, 160.0)
