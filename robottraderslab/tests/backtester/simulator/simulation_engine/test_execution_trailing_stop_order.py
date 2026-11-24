import math
from datetime import datetime

import pytest
from simulation_fixture import enter_position

from robottraderslab.backtester.simulator import FeeRates
from robottraderslab.exceptions import BotCriticalError, NoOpenPositionError
from robottraderslab.exchanges import OrderSide, StopLoss
from robottraderslab_core.exchanges import OrderType

ENTRY_PRICE = 100.0
TRAILING_SHARE = 0.1


def _trailing_stops(sim):
    return [order for order in sim.open_orders if order.kind == "trailing-stop"]


def _fired_prices(sim) -> list[float]:
    return [
        execution.price
        for execution in sim.simulation_engine.get_executions_since(datetime.min)
        if execution.kind is OrderType.TRAILING_STOP
    ]


class TestTrailingStopOrderExecution:
    async def test_a_long_trailing_stop_rests_a_share_below_the_last_price(
        self, sim, btc_usdt_perp
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY)

        await sim.exchange.update_position_trailing_stop_ratio(
            btc_usdt_perp, TRAILING_SHARE
        )

        (trailing_stop,) = _trailing_stops(sim)
        assert trailing_stop.trigger_price == pytest.approx(90.0)
        assert trailing_stop.side is OrderSide.SELL
        assert trailing_stop.reduce_only is True
        assert trailing_stop.quantity is None

    async def test_a_short_trailing_stop_rests_a_share_above_the_last_price(
        self, sim, btc_usdt_perp
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.SELL)

        await sim.exchange.update_position_trailing_stop_ratio(
            btc_usdt_perp, TRAILING_SHARE
        )

        (trailing_stop,) = _trailing_stops(sim)
        assert trailing_stop.trigger_price == pytest.approx(110.0)
        assert trailing_stop.side is OrderSide.BUY

    async def test_a_long_closes_within_the_candle_that_rises_then_falls(
        self, sim, btc_usdt_perp
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY)
        await sim.exchange.update_position_trailing_stop_ratio(
            btc_usdt_perp, TRAILING_SHARE
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=100.0, high=120.0, low=100.0, close=105.0
        )

        assert sim.open_positions == {}
        assert _trailing_stops(sim) == []
        assert _fired_prices(sim) == pytest.approx([108.0])

    async def test_a_short_closes_within_the_candle_that_falls_then_rises(
        self, sim, btc_usdt_perp
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.SELL)
        await sim.exchange.update_position_trailing_stop_ratio(
            btc_usdt_perp, TRAILING_SHARE
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=100.0, high=100.0, low=80.0, close=95.0
        )

        assert sim.open_positions == {}
        assert _fired_prices(sim) == pytest.approx([88.0])

    async def test_the_stop_keeps_the_level_the_best_candle_gave_it(
        self, sim, btc_usdt_perp
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY)
        await sim.exchange.update_position_trailing_stop_ratio(
            btc_usdt_perp, TRAILING_SHARE
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=100.0, high=120.0, low=109.0, close=110.0
        )
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=110.0, high=111.0, low=109.0, close=110.0
        )

        (trailing_stop,) = _trailing_stops(sim)
        assert trailing_stop.trigger_price == pytest.approx(108.0)
        assert len(sim.open_positions) == 1

    async def test_a_long_gapping_through_the_level_closes_at_the_open(
        self, sim, btc_usdt_perp
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY)
        await sim.exchange.update_position_trailing_stop_ratio(
            btc_usdt_perp, TRAILING_SHARE
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=80.0, high=130.0, low=75.0, close=85.0
        )

        assert _fired_prices(sim) == pytest.approx([80.0])

    async def test_a_short_gapping_through_the_level_closes_at_the_open(
        self, sim, btc_usdt_perp
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.SELL)
        await sim.exchange.update_position_trailing_stop_ratio(
            btc_usdt_perp, TRAILING_SHARE
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=120.0, high=125.0, low=70.0, close=115.0
        )

        assert _fired_prices(sim) == pytest.approx([120.0])

    async def test_a_candle_without_a_price_leaves_the_trailing_stop_resting(
        self, sim, btc_usdt_perp
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY)
        await sim.exchange.update_position_trailing_stop_ratio(
            btc_usdt_perp, TRAILING_SHARE
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp,
            open=math.nan,
            high=math.nan,
            low=math.nan,
            close=math.nan,
        )

        assert len(_trailing_stops(sim)) == 1
        assert len(sim.open_positions) == 1

    async def test_a_trailing_stop_closes_the_size_it_was_given(
        self, sim, btc_usdt_perp
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY, quantity=2.0)
        await sim.exchange.place_trailing_stop_order(
            btc_usdt_perp, distance=10.0, quantity=1.0
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=100.0, high=100.0, low=90.0, close=95.0
        )

        assert sim.open_positions[btc_usdt_perp].quantity == 1.0
        assert _trailing_stops(sim) == []

    async def test_a_stop_loss_closing_the_position_takes_the_trailing_stop_with_it(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            stop_loss=StopLoss(trigger_price=95.0),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=ENTRY_PRICE)
        await sim.exchange.update_position_trailing_stop_ratio(btc_usdt_perp, 0.5)

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=100.0, high=100.0, low=94.0, close=94.0
        )

        assert sim.open_positions == {}
        assert sim.open_orders == []

    async def test_the_fill_is_reported_as_a_trailing_stop(self, sim, btc_usdt_perp):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY)
        await sim.exchange.update_position_trailing_stop_ratio(
            btc_usdt_perp, TRAILING_SHARE
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=100.0, high=100.0, low=85.0, close=86.0
        )

        assert sim.fill_recorder.get_fills()["reason"].iloc[-1] == "trailing-stop"
        assert _fired_prices(sim) == pytest.approx([90.0])


class TestPlacingATrailingStop:
    async def test_a_price_distance_trails_the_share_of_the_last_price_it_makes(
        self, sim, btc_usdt_perp
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY)

        await sim.exchange.place_trailing_stop_order(btc_usdt_perp, distance=5.0)

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=100.0, high=200.0, low=195.0, close=196.0
        )
        (trailing_stop,) = _trailing_stops(sim)
        assert trailing_stop.trigger_price == pytest.approx(190.0)

    async def test_the_new_trailing_stop_replaces_the_one_resting(
        self, sim, btc_usdt_perp
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY)
        first = await sim.exchange.update_position_trailing_stop_ratio(
            btc_usdt_perp, TRAILING_SHARE
        )

        second = await sim.exchange.update_position_trailing_stop_ratio(
            btc_usdt_perp, 0.2
        )

        (trailing_stop,) = _trailing_stops(sim)
        assert trailing_stop.order_id == second.order_id
        assert first.order_id != second.order_id

    async def test_a_price_distance_replaces_the_one_resting(self, sim, btc_usdt_perp):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY)
        await sim.exchange.update_position_trailing_stop_ratio(
            btc_usdt_perp, TRAILING_SHARE
        )

        replacing = await sim.exchange.update_position_trailing_stop(
            btc_usdt_perp, distance=20.0
        )

        (trailing_stop,) = _trailing_stops(sim)
        assert trailing_stop.order_id == replacing.order_id
        assert trailing_stop.trigger_price == pytest.approx(80.0)

    async def test_no_open_position_to_trail(self, sim, btc_usdt_perp):
        with pytest.raises(NoOpenPositionError):
            await sim.exchange.update_position_trailing_stop_ratio(
                btc_usdt_perp, TRAILING_SHARE
            )

    async def test_a_trail_placed_with_an_activation_price(self, sim, btc_usdt_perp):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY)

        with pytest.raises(BotCriticalError, match="activation price"):
            await sim.exchange.place_trailing_stop_order(
                btc_usdt_perp, distance=5.0, activation_price=110.0
            )

        assert _trailing_stops(sim) == []

    @pytest.mark.parametrize("share", [0.0, 1.0, -0.1, 1.5])
    async def test_a_share_of_the_price_outside_zero_and_one(
        self, sim, btc_usdt_perp, share
    ):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY)

        with pytest.raises(BotCriticalError, match="within"):
            await sim.exchange.update_position_trailing_stop_ratio(btc_usdt_perp, share)

        assert _trailing_stops(sim) == []


class TestTrailingStopFee:
    @pytest.fixture
    def fee_rates(self) -> FeeRates:
        return FeeRates(maker=0.01, taker=0.02)

    async def test_the_fill_pays_the_taker_fee(self, sim, btc_usdt_perp):
        await enter_position(sim, btc_usdt_perp, OrderSide.BUY)
        await sim.exchange.update_position_trailing_stop_ratio(
            btc_usdt_perp, TRAILING_SHARE
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=100.0, high=100.0, low=85.0, close=86.0
        )

        (trailing_fill,) = (
            execution
            for execution in sim.simulation_engine.get_executions_since(datetime.min)
            if execution.kind is OrderType.TRAILING_STOP
        )
        assert trailing_fill.fee == pytest.approx(
            trailing_fill.quantity * trailing_fill.price * 0.02
        )
