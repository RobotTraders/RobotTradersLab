from datetime import datetime, timedelta

import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import OrderSide, StopLoss, TakeProfit

LONG_ENTRY_SL = 150.0
LONG_ENTRY_TP = 250.0
SHORT_ENTRY_SL = 210.0


async def _enter_long_with_stop_loss(sim, symbol: Symbol) -> None:
    await sim.exchange.place_market_order(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=1.0,
        stop_loss=StopLoss(trigger_price=LONG_ENTRY_SL),
    )


async def _enter_long_with_take_profit(sim, symbol: Symbol) -> None:
    await sim.exchange.place_market_order(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=1.0,
        take_profit=TakeProfit(trigger_price=LONG_ENTRY_TP),
    )


async def _enter_short_with_stop_loss(sim, symbol: Symbol) -> None:
    await sim.exchange.place_market_order(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=1.0,
        stop_loss=StopLoss(trigger_price=SHORT_ENTRY_SL),
    )


def _trigger_long_sl(sim, symbol: Symbol) -> None:
    sim.simulate_on_current_ohlcvs(
        symbol, low=LONG_ENTRY_SL - 10, high=200.0, close=160.0
    )


def _trigger_long_tp(sim, symbol: Symbol) -> None:
    sim.simulate_on_current_ohlcvs(
        symbol, low=180.0, high=LONG_ENTRY_TP + 10, close=240.0
    )


def _trigger_short_sl(sim, symbol: Symbol) -> None:
    sim.simulate_on_current_ohlcvs(
        symbol, low=180.0, high=SHORT_ENTRY_SL + 10, close=205.0
    )


class TestGetExecutionsSince:
    async def test_returns_empty_when_no_exits_fired(self, sim, btc_usdt_perp):
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, low=180.0, high=220.0, close=200.0
        )

        executions = await sim.exchange.get_executions_since(datetime.min, [])

        assert executions == []

    async def test_records_closing_side_when_long_stop_loss_fires(
        self, sim, btc_usdt_perp
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, open=200.0, close=200.0)
        await _enter_long_with_stop_loss(sim, btc_usdt_perp)
        _trigger_long_sl(sim, btc_usdt_perp)

        executions = await sim.exchange.get_executions_since(datetime.min, [])

        assert len(executions) == 2
        assert executions[-1].symbol == btc_usdt_perp
        assert executions[-1].side == OrderSide.SELL
        assert executions[-1].kind == "stop-loss"

    async def test_records_closing_side_when_short_stop_loss_fires(
        self, sim, btc_usdt_perp
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, open=200.0, close=200.0)
        await _enter_short_with_stop_loss(sim, btc_usdt_perp)
        _trigger_short_sl(sim, btc_usdt_perp)

        executions = await sim.exchange.get_executions_since(datetime.min, [])

        assert len(executions) == 2
        assert executions[-1].symbol == btc_usdt_perp
        assert executions[-1].side == OrderSide.BUY
        assert executions[-1].kind == "stop-loss"

    async def test_realised_profit_and_fee_are_stated_when_a_stop_loss_fires(
        self, sim, btc_usdt_perp
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, open=200.0, close=200.0)
        await _enter_long_with_stop_loss(sim, btc_usdt_perp)
        _trigger_long_sl(sim, btc_usdt_perp)

        executions = await sim.exchange.get_executions_since(datetime.min, [])

        assert executions[-1].realised_profit is not None
        assert executions[-1].fee is not None

    async def test_records_take_profit_fill(self, sim, btc_usdt_perp):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, open=200.0, close=200.0)
        await _enter_long_with_take_profit(sim, btc_usdt_perp)
        _trigger_long_tp(sim, btc_usdt_perp)

        executions = await sim.exchange.get_executions_since(datetime.min, [])

        assert len(executions) == 2
        assert executions[-1].symbol == btc_usdt_perp
        assert executions[-1].kind == "take-profit"

    async def test_returns_fills_across_symbols(
        self, sim, btc_usdt_perp, eth_usdt_perp
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, open=200.0, close=200.0)
        sim.simulate_on_current_ohlcvs(eth_usdt_perp, open=200.0, close=200.0)
        await _enter_long_with_stop_loss(sim, btc_usdt_perp)
        await _enter_long_with_stop_loss(sim, eth_usdt_perp)
        _trigger_long_sl(sim, btc_usdt_perp)
        _trigger_long_sl(sim, eth_usdt_perp)

        executions = await sim.exchange.get_executions_since(datetime.min, [])

        assert {execution.symbol for execution in executions} == {
            btc_usdt_perp,
            eth_usdt_perp,
        }

    async def test_includes_fills_at_the_since_boundary(self, sim, btc_usdt_perp):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, open=200.0, close=200.0)
        await _enter_long_with_stop_loss(sim, btc_usdt_perp)
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, low=180.0, high=220.0, close=200.0
        )
        _trigger_long_sl(sim, btc_usdt_perp)

        all_executions = await sim.exchange.get_executions_since(datetime.min, [])
        fill_time = all_executions[-1].timestamp

        at_fill_time = await sim.exchange.get_executions_since(fill_time, [])
        just_after = await sim.exchange.get_executions_since(
            fill_time + timedelta(microseconds=1), []
        )

        assert len(at_fill_time) == 1
        assert just_after == []


class TestRecordingIsGated:
    async def test_off_by_default(self, sim_without_recording, btc_usdt_perp):
        await sim_without_recording.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=1.0
        )
        sim_without_recording.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        assert (
            sim_without_recording.simulation_engine.get_executions_since(datetime.min)
            == []
        )


class TestRecordsEveryFillKind:
    entry_price = 100.0
    exit_price = 120.0

    async def test_records_a_plain_market_entry(self, sim, btc_usdt_perp):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            client_order_id="run-1-alpha",
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.entry_price)

        executions = await sim.exchange.get_executions_since(datetime.min, [])

        assert len(executions) == 1
        assert executions[0].kind == "market"
        assert executions[0].side == OrderSide.BUY
        assert executions[0].price == self.entry_price
        assert executions[0].realised_profit == 0.0
        assert executions[0].client_order_id == "run-1-alpha"

    async def test_records_a_plain_limit_entry(self, sim, btc_usdt_perp):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            price=self.entry_price,
        )
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp,
            open=self.entry_price,
            low=self.entry_price - 5,
            high=self.entry_price + 5,
        )

        executions = await sim.exchange.get_executions_since(datetime.min, [])

        assert len(executions) == 1
        assert executions[0].kind == "limit"
        assert executions[0].price == self.entry_price

    async def test_records_a_plain_exit(self, sim, btc_usdt_perp):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=2.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.entry_price)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.exit_price)
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=2.0,
            reduce_only=True,
        )

        executions = await sim.exchange.get_executions_since(datetime.min, [])

        assert len(executions) == 2
        assert executions[-1].kind == "market"
        assert executions[-1].side == OrderSide.SELL
        assert executions[-1].price == self.exit_price
        assert executions[-1].realised_profit > 0.0

    async def test_records_a_triggered_entry_as_trigger_kind(self, sim, btc_usdt_perp):
        trigger_price = 150.0
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            trigger_price=trigger_price,
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp,
            low=trigger_price - 10,
            high=trigger_price + 10,
            close=trigger_price,
        )

        executions = await sim.exchange.get_executions_since(datetime.min, [])

        assert len(executions) == 1
        assert executions[0].kind == "trigger"

    async def test_records_a_triggered_limit_entry_as_trigger_kind(
        self, sim, btc_usdt_perp
    ):
        trigger_price = 150.0
        limit_price = 140.0
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            price=limit_price,
            trigger_price=trigger_price,
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, open=155.0, low=145.0, high=155.0)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, open=145.0, low=135.0, high=145.0)

        executions = await sim.exchange.get_executions_since(datetime.min, [])

        assert len(executions) == 1
        assert executions[0].kind == "trigger"
        assert executions[0].price == limit_price

    async def test_records_both_legs_of_a_flipping_entry(self, sim, btc_usdt_perp):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=10.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.entry_price)

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=15.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.entry_price)

        executions = await sim.exchange.get_executions_since(datetime.min, [])
        close_leg, open_leg = executions[-2], executions[-1]

        assert close_leg.side == OrderSide.SELL
        assert close_leg.quantity == 10.0
        assert close_leg.order_id == open_leg.order_id
        assert close_leg.execution_id != open_leg.execution_id
        assert open_leg.side == OrderSide.SELL
        assert open_leg.quantity == 5.0
        assert open_leg.realised_profit == 0.0
        assert open_leg.fee == 0.0

    @pytest.mark.parametrize(
        ("entry_side", "closing_side", "liquidating_low", "liquidating_high"),
        [
            (OrderSide.BUY, OrderSide.SELL, 39_000.0, 42_000.0),
            (OrderSide.SELL, OrderSide.BUY, 58_000.0, 61_000.0),
        ],
    )
    async def test_records_a_liquidation_with_no_caller_label(
        self,
        sim,
        btc_usdt_perp,
        entry_side,
        closing_side,
        liquidating_low,
        liquidating_high,
    ):
        sim.simulation_engine.set_symbol_leverage(btc_usdt_perp, 5.0)
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=entry_side,
            quantity=1.0,
            client_order_id="run-1-alpha",
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=50_000.0)

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, low=liquidating_low, high=liquidating_high, close=50_000.0
        )

        executions = await sim.exchange.get_executions_since(datetime.min, [])

        assert executions[-1].kind == "liquidation"
        assert executions[-1].side == closing_side
        assert executions[-1].client_order_id is None
        assert executions[-1].realised_profit is not None
        assert executions[-1].realised_profit < 0.0
