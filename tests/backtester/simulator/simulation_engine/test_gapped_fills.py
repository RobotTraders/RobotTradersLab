from datetime import datetime

import pytest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import FeeRates
from robottraderslab.exchanges import Execution, OrderSide, StopLoss, TakeProfit

MAKER = 0.001
TAKER = 0.005
QUANTITY = 2.0


@pytest.fixture
def fee_rates() -> FeeRates:
    return FeeRates(maker=MAKER, taker=TAKER)


async def _enter_with_protection(
    sim, symbol: Symbol, side: OrderSide, stop_loss: float, take_profit: float
) -> None:
    sim.simulate_on_current_ohlcvs(symbol, close=95.0)
    await sim.exchange.place_market_order(
        symbol=symbol,
        side=side,
        quantity=QUANTITY,
        stop_loss=StopLoss(trigger_price=stop_loss),
        take_profit=TakeProfit(trigger_price=take_profit),
    )


async def _place_trigger_entry(
    sim, symbol: Symbol, side: OrderSide, placement_close: float
) -> None:
    sim.simulate_on_current_ohlcvs(symbol, close=placement_close)
    await sim.exchange.place_market_order(
        symbol=symbol, side=side, quantity=QUANTITY, trigger_price=100.0
    )


async def _place_limit_entry(sim, symbol: Symbol, side: OrderSide) -> None:
    await sim.exchange.place_limit_order(
        symbol=symbol, side=side, quantity=QUANTITY, price=100.0
    )


async def _last_execution(sim) -> Execution:
    executions = await sim.exchange.get_executions_since(datetime.min, [])
    return executions[-1]


class TestProtection:
    @pytest.mark.parametrize(
        ("side", "take_profit", "candle", "exit_price"),
        [
            (OrderSide.BUY, 100.0, {"open": 105.0, "low": 103.0, "high": 106.0}, 105.0),
            (OrderSide.SELL, 90.0, {"open": 85.0, "low": 84.0, "high": 87.0}, 85.0),
        ],
    )
    async def test_take_profit_exits_at_the_open(
        self, sim, btc_usdt_perp, side, take_profit, candle, exit_price
    ):
        await _enter_with_protection(
            sim,
            btc_usdt_perp,
            side,
            stop_loss=50.0 if side == OrderSide.BUY else 150.0,
            take_profit=take_profit,
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=candle["open"], **candle)

        assert sim.open_positions == {}
        assert (await _last_execution(sim)).price == exit_price

    @pytest.mark.parametrize(
        ("side", "stop_loss", "candle", "exit_price"),
        [
            (OrderSide.BUY, 90.0, {"open": 85.0, "low": 84.0, "high": 86.0}, 85.0),
            (
                OrderSide.SELL,
                100.0,
                {"open": 105.0, "low": 104.0, "high": 106.0},
                105.0,
            ),
        ],
    )
    async def test_stop_loss_exits_at_the_open(
        self, sim, btc_usdt_perp, side, stop_loss, candle, exit_price
    ):
        await _enter_with_protection(
            sim,
            btc_usdt_perp,
            side,
            stop_loss=stop_loss,
            take_profit=150.0 if side == OrderSide.BUY else 50.0,
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=candle["open"], **candle)

        assert sim.open_positions == {}
        assert (await _last_execution(sim)).price == exit_price

    @pytest.mark.parametrize(
        ("side", "stop_loss", "take_profit", "candle", "exit_price"),
        [
            (
                OrderSide.BUY,
                90.0,
                100.0,
                {"open": 97.0, "low": 96.0, "high": 101.0},
                100.0,
            ),
            (
                OrderSide.BUY,
                90.0,
                100.0,
                {"open": 93.0, "low": 89.0, "high": 94.0},
                90.0,
            ),
            (
                OrderSide.SELL,
                100.0,
                90.0,
                {"open": 93.0, "low": 89.0, "high": 94.0},
                90.0,
            ),
            (
                OrderSide.SELL,
                100.0,
                90.0,
                {"open": 97.0, "low": 96.0, "high": 101.0},
                100.0,
            ),
        ],
    )
    async def test_level_inside_the_candle_exits_at_the_level(
        self, sim, btc_usdt_perp, side, stop_loss, take_profit, candle, exit_price
    ):
        await _enter_with_protection(sim, btc_usdt_perp, side, stop_loss, take_profit)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=candle["open"], **candle)

        assert sim.open_positions == {}
        assert (await _last_execution(sim)).price == exit_price


class TestTriggerEntry:
    @pytest.mark.parametrize(
        ("side", "placement_close", "candle", "fill_price"),
        [
            (OrderSide.BUY, 102.0, {"open": 97.0, "low": 96.0, "high": 99.0}, 97.0),
            (OrderSide.SELL, 98.0, {"open": 103.0, "low": 101.0, "high": 104.0}, 103.0),
        ],
    )
    async def test_gapped_through_fires_at_the_open(
        self, sim, btc_usdt_perp, side, placement_close, candle, fill_price
    ):
        await _place_trigger_entry(sim, btc_usdt_perp, side, placement_close)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=candle["open"], **candle)

        assert sim.open_positions[btc_usdt_perp].average_entry_price == fill_price

    @pytest.mark.parametrize(
        ("side", "placement_close", "candle"),
        [
            (OrderSide.BUY, 98.0, {"open": 97.0, "low": 96.0, "high": 99.0}),
            (OrderSide.SELL, 102.0, {"open": 103.0, "low": 101.0, "high": 104.0}),
        ],
    )
    async def test_level_not_reached_keeps_it_waiting(
        self, sim, btc_usdt_perp, side, placement_close, candle
    ):
        await _place_trigger_entry(sim, btc_usdt_perp, side, placement_close)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=candle["open"], **candle)

        assert sim.open_positions == {}
        assert sim.open_orders[0].kind == "trigger"

    @pytest.mark.parametrize(
        ("side", "placement_close", "candle"),
        [
            (OrderSide.BUY, 98.0, {"open": 99.0, "low": 98.5, "high": 101.0}),
            (OrderSide.SELL, 102.0, {"open": 101.0, "low": 99.0, "high": 101.5}),
        ],
    )
    async def test_level_inside_the_candle_fires_at_the_level(
        self, sim, btc_usdt_perp, side, placement_close, candle
    ):
        await _place_trigger_entry(sim, btc_usdt_perp, side, placement_close)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=candle["open"], **candle)

        assert sim.open_positions[btc_usdt_perp].average_entry_price == 100.0

    async def test_placed_with_no_close_fires_only_inside_its_first_candle(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            trigger_price=100.0,
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=97.0, low=96.0, high=99.0, close=98.0
        )

        assert sim.open_positions == {}

    async def test_placed_with_no_close_waits_on_the_side_of_its_first_close(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            trigger_price=100.0,
        )
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=97.0, low=96.0, high=99.0, close=98.0
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=103.0, low=102.0, high=104.0, close=103.0
        )

        assert sim.open_positions[btc_usdt_perp].average_entry_price == 103.0

    async def test_stop_limit_fired_inside_the_candle_fills_at_its_level(
        self, sim, btc_usdt_perp
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=98.0)
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            price=101.0,
            trigger_price=100.0,
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=99.0, low=98.5, high=102.0, close=101.5
        )

        assert sim.open_positions[btc_usdt_perp].average_entry_price == 100.0


class TestRestingLimit:
    @pytest.mark.parametrize(
        ("side", "candle", "fill_price"),
        [
            (OrderSide.BUY, {"open": 97.0, "low": 96.0, "high": 99.0}, 97.0),
            (OrderSide.SELL, {"open": 103.0, "low": 101.0, "high": 104.0}, 103.0),
        ],
    )
    async def test_gapped_through_fills_at_the_open(
        self, sim, btc_usdt_perp, side, candle, fill_price
    ):
        await _place_limit_entry(sim, btc_usdt_perp, side)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=candle["open"], **candle)

        assert sim.open_positions[btc_usdt_perp].average_entry_price == fill_price

    async def test_gapped_through_pays_the_taker_rate(self, sim, btc_usdt_perp):
        await _place_limit_entry(sim, btc_usdt_perp, OrderSide.BUY)

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=97.0, low=96.0, high=99.0, close=98.0
        )

        assert (await _last_execution(sim)).fee == pytest.approx(
            QUANTITY * 97.0 * TAKER
        )

    @pytest.mark.parametrize(
        ("side", "candle"),
        [
            (OrderSide.BUY, {"open": 102.0, "low": 99.0, "high": 103.0}),
            (OrderSide.SELL, {"open": 98.0, "low": 97.0, "high": 101.0}),
        ],
    )
    async def test_level_inside_the_candle_fills_at_the_level_as_maker(
        self, sim, btc_usdt_perp, side, candle
    ):
        await _place_limit_entry(sim, btc_usdt_perp, side)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=candle["open"], **candle)

        execution = await _last_execution(sim)
        assert execution.price == 100.0
        assert execution.fee == pytest.approx(QUANTITY * 100.0 * MAKER)
