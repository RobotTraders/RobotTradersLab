from robottraderslab import Symbol
from robottraderslab.exchanges import OrderSide, PositionSide, StopLoss, TakeProfit


class TestTakeProfitOrderExecution:
    take_profit = 150.0
    stop_loss_long = 100.0
    stop_loss_short = 200.0

    async def place_enter_long_with_take_profit(
        self,
        sim,
        btc_usdt_perp: Symbol,
        with_stop_loss: bool = False,
    ):
        stop_loss = (
            StopLoss(trigger_price=self.stop_loss_long) if with_stop_loss else None
        )
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            stop_loss=stop_loss,
            take_profit=TakeProfit(trigger_price=self.take_profit),
        )

    async def place_enter_short_with_take_profit(
        self,
        sim,
        btc_usdt_perp: Symbol,
        with_stop_loss: bool = False,
    ):
        stop_loss = (
            StopLoss(trigger_price=self.stop_loss_short) if with_stop_loss else None
        )
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=2.0,
            stop_loss=stop_loss,
            take_profit=TakeProfit(trigger_price=self.take_profit),
        )

    def simulate_filled_take_profit_order(
        self, sim, btc_usdt_perp: Symbol, side: PositionSide
    ):
        ohlcv = (
            {
                "low": self.take_profit - 20,
                "high": self.take_profit + 20,
                "close": self.take_profit - 10,
            }
            if side == PositionSide.LONG
            else {
                "low": self.take_profit - 20,
                "high": self.take_profit + 20,
                "close": self.take_profit + 10,
            }
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, **ohlcv)

    def simulate_unfilled_tpsl_orders(
        self, sim, btc_usdt_perp: Symbol, side: PositionSide
    ):
        ohlcv = (
            {
                "low": self.take_profit - 20,
                "high": self.take_profit - 10,
                "close": self.take_profit - 10,
            }
            if side == PositionSide.LONG
            else {
                "low": self.take_profit + 10,
                "high": self.take_profit + 20,
                "close": self.take_profit + 10,
            }
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, **ohlcv)

    async def test_long_take_profit_not_yet_executed(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_long_with_take_profit(sim, btc_usdt_perp)

        self.simulate_unfilled_tpsl_orders(sim, btc_usdt_perp, PositionSide.LONG)

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 1
        assert sim.open_orders[0].kind == "take-profit"
        assert sim.simulation_engine.get_equity("USDT") == sim.initial_usdt

    async def test_short_take_profit_not_yet_executed(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_short_with_take_profit(sim, btc_usdt_perp)

        self.simulate_unfilled_tpsl_orders(sim, btc_usdt_perp, PositionSide.SHORT)

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 1
        assert sim.open_orders[0].kind == "take-profit"
        assert sim.simulation_engine.get_equity("USDT") == sim.initial_usdt

    async def test_long_take_profit_executed(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_long_with_take_profit(sim, btc_usdt_perp)

        self.simulate_filled_take_profit_order(sim, btc_usdt_perp, PositionSide.LONG)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0
        assert sim.simulation_engine.get_equity("USDT") > sim.initial_usdt

    async def test_short_take_profit_executed(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_short_with_take_profit(sim, btc_usdt_perp)

        self.simulate_filled_take_profit_order(sim, btc_usdt_perp, PositionSide.SHORT)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0
        assert sim.simulation_engine.get_equity("USDT") > sim.initial_usdt

    async def test_long_tpsl_take_profit_executed(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_long_with_take_profit(
            sim, btc_usdt_perp, with_stop_loss=True
        )

        self.simulate_unfilled_tpsl_orders(sim, btc_usdt_perp, PositionSide.LONG)
        assert len(sim.open_orders) == 2
        assert len(sim.open_positions) == 1
        assert sim.open_orders[0].kind == "stop-loss"
        assert sim.open_orders[1].kind == "take-profit"

        self.simulate_filled_take_profit_order(sim, btc_usdt_perp, PositionSide.LONG)
        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0
        assert sim.simulation_engine.get_equity("USDT") > sim.initial_usdt

    async def test_short_tpsl_take_profit_executed(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_short_with_take_profit(
            sim, btc_usdt_perp, with_stop_loss=True
        )

        self.simulate_unfilled_tpsl_orders(sim, btc_usdt_perp, PositionSide.SHORT)
        assert len(sim.open_orders) == 2
        assert len(sim.open_positions) == 1
        assert sim.open_orders[0].kind == "stop-loss"
        assert sim.open_orders[1].kind == "take-profit"

        self.simulate_filled_take_profit_order(sim, btc_usdt_perp, PositionSide.SHORT)
        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0
        assert sim.simulation_engine.get_equity("USDT") > sim.initial_usdt
