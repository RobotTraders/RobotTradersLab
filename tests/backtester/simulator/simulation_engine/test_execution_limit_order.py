import pytest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import FeeRates
from robottraderslab.exceptions import ExchangeRecoverableError
from robottraderslab.exchanges import OrderSide, PositionSide, TimeInForce


class TestLimitOrderExecution:
    quantity = 2.0
    limit_price = 100.0

    async def place_enter_long(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=self.quantity,
            price=self.limit_price,
        )

    async def place_enter_short(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=self.quantity,
            price=self.limit_price,
        )

    async def place_exit_long(self, sim, btc_usdt_perp: Symbol, quantity: float = 2.0):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=quantity,
            price=self.limit_price,
            reduce_only=True,
        )

    async def place_exit_short(self, sim, btc_usdt_perp: Symbol, quantity: float = 2.0):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=quantity,
            price=self.limit_price,
            reduce_only=True,
        )

    def simulate_filled_limit_order(self, sim, btc_usdt_perp: Symbol):
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp,
            low=self.limit_price - 10,
            high=self.limit_price + 20,
            close=self.limit_price + 15,
        )

    def simulate_unfilled_limit_order(self, sim, btc_usdt_perp: Symbol):
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp,
            low=self.limit_price + 10,
            high=self.limit_price + 20,
            close=self.limit_price + 15,
        )

    async def test_enter_long_not_yet_executed(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_long(sim, btc_usdt_perp)

        self.simulate_unfilled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0

    async def test_enter_short_not_yet_executed(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_short(sim, btc_usdt_perp)

        self.simulate_unfilled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 0

    async def test_enter_long_executed_at_limit_price(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_long(sim, btc_usdt_perp)

        self.simulate_filled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.open_positions[btc_usdt_perp].quantity == self.quantity
        assert sim.open_positions[btc_usdt_perp].average_entry_price == self.limit_price
        assert sim.open_positions[btc_usdt_perp].side == PositionSide.LONG

    async def test_enter_short_executed_at_limit_price(
        self, sim, btc_usdt_perp: Symbol
    ):
        await self.place_enter_short(sim, btc_usdt_perp)

        self.simulate_filled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.open_positions[btc_usdt_perp].quantity == self.quantity
        assert sim.open_positions[btc_usdt_perp].average_entry_price == self.limit_price
        assert sim.open_positions[btc_usdt_perp].side == PositionSide.SHORT

    async def test_exit_long_without_position(self, sim, btc_usdt_perp: Symbol):
        await self.place_exit_long(sim, btc_usdt_perp)

        self.simulate_filled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0

    async def test_exit_short_without_position(self, sim, btc_usdt_perp: Symbol):
        await self.place_exit_short(sim, btc_usdt_perp)

        self.simulate_filled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0

    async def test_exit_long_not_yet_executed(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_long(sim, btc_usdt_perp)
        self.simulate_filled_limit_order(sim, btc_usdt_perp)
        await self.place_exit_long(sim, btc_usdt_perp)

        self.simulate_unfilled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 1

    async def test_exit_short_not_yet_executed(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_short(sim, btc_usdt_perp)
        self.simulate_filled_limit_order(sim, btc_usdt_perp)
        await self.place_exit_short(sim, btc_usdt_perp)

        self.simulate_unfilled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 1
        assert len(sim.open_positions) == 1

    async def test_exit_long_with_full_position(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_long(sim, btc_usdt_perp)
        self.simulate_filled_limit_order(sim, btc_usdt_perp)
        await self.place_exit_long(sim, btc_usdt_perp)

        self.simulate_filled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0

    async def test_exit_short_with_full_position(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_short(sim, btc_usdt_perp)
        self.simulate_filled_limit_order(sim, btc_usdt_perp)
        await self.place_exit_short(sim, btc_usdt_perp)

        self.simulate_filled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 0

    async def test_increase_long_position(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_long(sim, btc_usdt_perp)
        self.simulate_filled_limit_order(sim, btc_usdt_perp)
        await self.place_enter_long(sim, btc_usdt_perp)

        self.simulate_filled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.open_positions[btc_usdt_perp].quantity == self.quantity * 2
        assert sim.open_positions[btc_usdt_perp].side == PositionSide.LONG

    async def test_increase_short_position(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_short(sim, btc_usdt_perp)
        self.simulate_filled_limit_order(sim, btc_usdt_perp)
        await self.place_enter_short(sim, btc_usdt_perp)

        self.simulate_filled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.open_positions[btc_usdt_perp].quantity == self.quantity * 2
        assert sim.open_positions[btc_usdt_perp].side == PositionSide.SHORT

    async def test_reduce_long_position(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_long(sim, btc_usdt_perp)
        self.simulate_filled_limit_order(sim, btc_usdt_perp)
        await self.place_exit_long(sim, btc_usdt_perp, self.quantity / 2)

        self.simulate_filled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.open_positions[btc_usdt_perp].quantity == self.quantity / 2
        assert sim.open_positions[btc_usdt_perp].side == PositionSide.LONG

    async def test_reduce_short_position(self, sim, btc_usdt_perp: Symbol):
        await self.place_enter_short(sim, btc_usdt_perp)
        self.simulate_filled_limit_order(sim, btc_usdt_perp)
        await self.place_exit_short(sim, btc_usdt_perp, self.quantity / 2)

        self.simulate_filled_limit_order(sim, btc_usdt_perp)

        assert len(sim.open_orders) == 0
        assert len(sim.open_positions) == 1
        assert sim.open_positions[btc_usdt_perp].quantity == self.quantity / 2
        assert sim.open_positions[btc_usdt_perp].side == PositionSide.SHORT


class TestTimeInForce:
    quantity = 2.0

    async def place_long_entry(self, sim, btc_usdt_perp: Symbol, time_in_force, price):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=self.quantity,
            price=price,
            time_in_force=time_in_force,
        )

    async def test_ioc_at_or_above_the_close_fills_at_the_close(
        self, sim, btc_usdt_perp
    ):
        await self.place_long_entry(sim, btc_usdt_perp, TimeInForce.IOC, 105.0)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=95.0, high=110.0, close=100.0)

        assert sim.open_orders == []
        assert sim.open_positions[btc_usdt_perp].average_entry_price == 100.0

    async def test_ioc_placed_after_a_candle_fills_at_its_close(
        self, sim, btc_usdt_perp
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=95.0, high=110.0, close=100.0)

        await self.place_long_entry(sim, btc_usdt_perp, TimeInForce.IOC, 105.0)

        assert sim.open_orders == []
        assert sim.open_positions[btc_usdt_perp].average_entry_price == 100.0

    async def test_ioc_below_the_close_is_cancelled(self, sim, btc_usdt_perp):
        await self.place_long_entry(sim, btc_usdt_perp, TimeInForce.IOC, 90.0)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=85.0, high=110.0, close=100.0)

        assert sim.open_orders == []
        assert sim.open_positions == {}
        assert sim.simulation_engine.get_balances()["USDT"].locked == 0.0

    @pytest.mark.parametrize("price", [100.0, 105.0])
    async def test_post_only_at_or_above_the_close_is_refused(
        self, sim, btc_usdt_perp, price
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=95.0, high=110.0, close=100.0)

        with pytest.raises(ExchangeRecoverableError, match="would take liquidity"):
            await self.place_long_entry(
                sim, btc_usdt_perp, TimeInForce.POST_ONLY, price
            )

        assert sim.open_orders == []
        assert sim.simulation_engine.get_balances()["USDT"].locked == 0.0

    @pytest.mark.parametrize("price", [100.0, 95.0])
    async def test_short_post_only_at_or_below_the_close_is_refused(
        self, sim, btc_usdt_perp, price
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=95.0, high=110.0, close=100.0)

        with pytest.raises(ExchangeRecoverableError, match="would take liquidity"):
            await sim.exchange.place_limit_order(
                symbol=btc_usdt_perp,
                side=OrderSide.SELL,
                quantity=self.quantity,
                price=price,
                time_in_force=TimeInForce.POST_ONLY,
            )

        assert sim.open_orders == []
        assert sim.simulation_engine.get_balances()["USDT"].locked == 0.0

    async def test_reduce_only_post_only_at_or_above_the_close_is_refused(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=self.quantity
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=95.0, high=110.0, close=100.0)

        with pytest.raises(ExchangeRecoverableError, match="would take liquidity"):
            await sim.exchange.place_limit_order(
                symbol=btc_usdt_perp,
                side=OrderSide.BUY,
                quantity=self.quantity,
                price=105.0,
                reduce_only=True,
                time_in_force=TimeInForce.POST_ONLY,
            )

        assert sim.open_orders == []
        assert sim.open_positions[btc_usdt_perp].quantity == self.quantity

    async def test_post_only_placed_on_a_gap_row_is_judged_at_the_next_close(
        self, sim, btc_usdt_perp, caplog
    ):
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, low=float("nan"), high=float("nan"), close=float("nan")
        )
        await self.place_long_entry(sim, btc_usdt_perp, TimeInForce.POST_ONLY, 105.0)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=95.0, high=110.0, close=100.0)

        assert sim.open_orders == []
        assert "would take liquidity" in caplog.text

    async def test_post_only_below_the_close_rests_when_placed_at_it(
        self, sim, btc_usdt_perp
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=95.0, high=110.0, close=100.0)

        await self.place_long_entry(sim, btc_usdt_perp, TimeInForce.POST_ONLY, 90.0)

        assert len(sim.open_orders) == 1

    async def test_post_only_placed_before_any_close_is_cancelled_at_the_first(
        self, sim, btc_usdt_perp, caplog
    ):
        await self.place_long_entry(sim, btc_usdt_perp, TimeInForce.POST_ONLY, 105.0)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=95.0, high=110.0, close=100.0)

        assert sim.open_orders == []
        assert sim.open_positions == {}
        assert sim.simulation_engine.get_balances()["USDT"].locked == 0.0
        assert "would take liquidity" in caplog.text

    async def test_post_only_below_the_close_rests_like_any_limit(
        self, sim, btc_usdt_perp
    ):
        await self.place_long_entry(sim, btc_usdt_perp, TimeInForce.POST_ONLY, 90.0)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=95.0, high=110.0, close=100.0)
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=80.0, high=95.0, close=85.0)

        assert sim.open_orders == []
        assert sim.open_positions[btc_usdt_perp].average_entry_price == 90.0

    async def test_post_only_below_the_close_fills_within_the_same_candle(
        self, sim, btc_usdt_perp
    ):
        await self.place_long_entry(sim, btc_usdt_perp, TimeInForce.POST_ONLY, 90.0)

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=85.0, high=110.0, close=100.0)

        assert sim.open_orders == []
        assert sim.open_positions[btc_usdt_perp].average_entry_price == 90.0

    async def test_short_ioc_at_or_below_the_close_fills(self, sim, btc_usdt_perp):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=self.quantity,
            price=95.0,
            time_in_force=TimeInForce.IOC,
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=90.0, high=110.0, close=100.0)

        assert sim.open_positions[btc_usdt_perp].side == PositionSide.SHORT
        assert sim.open_positions[btc_usdt_perp].average_entry_price == 100.0

    async def test_reduce_only_ioc_closes_the_position_at_the_close(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=self.quantity
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=self.quantity,
            price=95.0,
            reduce_only=True,
            time_in_force=TimeInForce.IOC,
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=90.0, high=110.0, close=100.0)

        assert sim.open_positions == {}
        assert sim.fill_recorder.get_fills()["price"].iloc[-1] == 100.0

    async def test_reduce_only_ioc_short_of_the_close_leaves_the_position(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=self.quantity
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=self.quantity,
            price=105.0,
            reduce_only=True,
            time_in_force=TimeInForce.IOC,
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=90.0, high=110.0, close=100.0)

        assert sim.open_orders == []
        assert sim.open_positions[btc_usdt_perp].quantity == self.quantity


class TestTimeInForceFee:
    quantity = 2.0

    @pytest.fixture
    def fee_rates(self) -> FeeRates:
        return FeeRates(maker=0.0002, taker=0.0006)

    async def test_ioc_pays_the_taker_fee(self, sim, btc_usdt_perp, fee_rates):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=self.quantity,
            price=105.0,
            time_in_force=TimeInForce.IOC,
        )
        before = sim.simulation_engine.get_balances()["USDT"].total

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=95.0, high=110.0, close=100.0)

        after = sim.simulation_engine.get_balances()["USDT"].total
        assert before - after == pytest.approx(self.quantity * 100.0 * fee_rates.taker)
