from robottraderslab import Symbol
from robottraderslab.exchanges import OrderSide, PositionSide, StopLoss, TakeProfit


class TestStopLossOrderExecution:
    stop_loss = 150.0
    take_profit_long = 200.0
    take_profit_short = 100.0

    async def place_enter_long_with_stop_loss(
        self,
        sim,
        btc_usdt_perp: Symbol,
        with_take_profit: bool = False,
    ):
        take_profit = (
            TakeProfit(trigger_price=self.take_profit_long)
            if with_take_profit
            else None
        )
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            stop_loss=StopLoss(trigger_price=self.stop_loss),
            take_profit=take_profit,
        )

    async def place_enter_short_with_stop_loss(
        self,
        sim,
        btc_usdt_perp: Symbol,
        with_take_profit: bool = False,
    ):
        take_profit = (
            TakeProfit(trigger_price=self.take_profit_short)
            if with_take_profit
            else None
        )
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=2.0,
            stop_loss=StopLoss(trigger_price=self.stop_loss),
            take_profit=take_profit,
        )

    def simulate_filled_stop_loss_order(
        self, sim, btc_usdt_perp: Symbol, side: PositionSide
    ):
        ohlcv = (
            {
                "low": self.stop_loss - 20,
                "high": self.stop_loss + 20,
                "close": self.stop_loss + 10,
            }
            if side == PositionSide.LONG
            else {
                "low": self.stop_loss - 20,
                "high": self.stop_loss + 20,
                "close": self.stop_loss - 10,
            }
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, **ohlcv)

    def simulate_filled_tpsl_orders(
        self, sim, btc_usdt_perp: Symbol, side: PositionSide
    ):
        ohlcv = (
            {
                "low": self.stop_loss - 20,
                "high": self.take_profit_long + 20,
                "close": self.stop_loss + 10,
            }
            if side == PositionSide.LONG
            else {
                "low": self.take_profit_short - 20,
                "high": self.stop_loss + 20,
                "close": self.stop_loss - 10,
            }
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, **ohlcv)

    def simulate_unfilled_tpsl_orders(
        self, sim, btc_usdt_perp: Symbol, side: PositionSide
    ):
        ohlcv = (
            {
                "low": self.stop_loss + 10,
                "high": self.stop_loss + 20,
                "close": self.stop_loss + 10,
            }
            if side == PositionSide.LONG
            else {
                "low": self.stop_loss - 20,
                "high": self.stop_loss - 10,
                "close": self.stop_loss - 10,
            }
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, **ohlcv)

    async def test_long_stop_loss_not_yet_executed(self, sim, btc_usdt_perp):
        await self.place_enter_long_with_stop_loss(sim, btc_usdt_perp)

        self.simulate_unfilled_tpsl_orders(sim, btc_usdt_perp, PositionSide.LONG)

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 1
        assert sim.open_orders[0].kind == "stop-loss"
        assert sim.simulation_engine.get_equity("USDT") == sim.initial_usdt

    async def test_short_stop_loss_not_yet_executed(self, sim, btc_usdt_perp):
        await self.place_enter_short_with_stop_loss(sim, btc_usdt_perp)

        self.simulate_unfilled_tpsl_orders(sim, btc_usdt_perp, PositionSide.SHORT)

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 1
        assert sim.open_orders[0].kind == "stop-loss"
        assert sim.simulation_engine.get_equity("USDT") == sim.initial_usdt

    async def test_long_stop_loss_executed(self, sim, btc_usdt_perp):
        await self.place_enter_long_with_stop_loss(sim, btc_usdt_perp)

        self.simulate_unfilled_tpsl_orders(sim, btc_usdt_perp, PositionSide.LONG)
        self.simulate_filled_stop_loss_order(sim, btc_usdt_perp, PositionSide.LONG)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0
        assert sim.simulation_engine.get_equity("USDT") < sim.initial_usdt

    async def test_short_stop_loss_executed(self, sim, btc_usdt_perp):
        await self.place_enter_short_with_stop_loss(sim, btc_usdt_perp)

        self.simulate_unfilled_tpsl_orders(sim, btc_usdt_perp, PositionSide.SHORT)
        self.simulate_filled_stop_loss_order(sim, btc_usdt_perp, PositionSide.SHORT)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0
        assert sim.simulation_engine.get_equity("USDT") < sim.initial_usdt

    async def test_long_tpsl_stop_loss_executed(self, sim, btc_usdt_perp):
        await self.place_enter_long_with_stop_loss(
            sim, btc_usdt_perp, with_take_profit=True
        )

        self.simulate_unfilled_tpsl_orders(sim, btc_usdt_perp, PositionSide.LONG)
        assert len(sim.open_orders) == 2
        assert len(sim.open_positions) == 1
        assert sim.open_orders[0].kind == "stop-loss"
        assert sim.open_orders[1].kind == "take-profit"

        self.simulate_filled_stop_loss_order(sim, btc_usdt_perp, PositionSide.LONG)
        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0
        assert sim.simulation_engine.get_equity("USDT") < sim.initial_usdt

    async def test_short_tpsl_stop_loss_executed(self, sim, btc_usdt_perp):
        await self.place_enter_short_with_stop_loss(
            sim, btc_usdt_perp, with_take_profit=True
        )

        self.simulate_unfilled_tpsl_orders(sim, btc_usdt_perp, PositionSide.SHORT)
        assert len(sim.open_orders) == 2
        assert len(sim.open_positions) == 1
        assert sim.open_orders[0].kind == "stop-loss"
        assert sim.open_orders[1].kind == "take-profit"

        self.simulate_filled_stop_loss_order(sim, btc_usdt_perp, PositionSide.SHORT)
        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0
        assert sim.simulation_engine.get_equity("USDT") < sim.initial_usdt

    async def test_long_stop_loss_wins_over_take_profit(self, sim, btc_usdt_perp):
        await self.place_enter_long_with_stop_loss(
            sim, btc_usdt_perp, with_take_profit=True
        )

        self.simulate_unfilled_tpsl_orders(sim, btc_usdt_perp, PositionSide.LONG)
        self.simulate_filled_tpsl_orders(sim, btc_usdt_perp, PositionSide.LONG)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0
        assert sim.simulation_engine.get_equity("USDT") < sim.initial_usdt

    async def test_short_stop_loss_wins_over_take_profit(self, sim, btc_usdt_perp):
        await self.place_enter_short_with_stop_loss(
            sim, btc_usdt_perp, with_take_profit=True
        )

        self.simulate_unfilled_tpsl_orders(sim, btc_usdt_perp, PositionSide.SHORT)
        self.simulate_filled_tpsl_orders(sim, btc_usdt_perp, PositionSide.SHORT)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0
        assert sim.simulation_engine.get_equity("USDT") < sim.initial_usdt
