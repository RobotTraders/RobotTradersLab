import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import (
    Balance,
    OrderSide,
    PositionSide,
    StopLoss,
    TakeProfit,
)


class TestLiquidation:
    entry_price = 50_000.0
    leverage = 5.0
    quantity = 1.0

    @pytest.fixture(autouse=True)
    def setup_leverage(self, sim, btc_usdt_perp: Symbol):
        sim.simulation_engine.set_symbol_leverage(btc_usdt_perp, self.leverage)

    @property
    def margin(self):
        return self.quantity * self.entry_price / self.leverage

    @pytest.mark.parametrize(
        ("side", "liquidating_low", "liquidating_high"),
        [
            (OrderSide.BUY, 39_000.0, 42_000.0),
            (OrderSide.SELL, 58_000.0, 61_000.0),
        ],
    )
    async def test_position_liquidated_when_price_crosses_threshold(
        self,
        sim,
        btc_usdt_perp: Symbol,
        side,
        liquidating_low,
        liquidating_high,
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=side, quantity=1.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.entry_price)

        assert len(sim.open_positions) == 1

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, low=liquidating_low, high=liquidating_high, close=50_000.0
        )

        assert len(sim.open_positions) == 0

    @pytest.mark.parametrize(
        ("side", "safe_low", "safe_high"),
        [
            (OrderSide.BUY, 41_000.0, 55_000.0),
            (OrderSide.SELL, 45_000.0, 59_000.0),
        ],
    )
    async def test_position_not_liquidated_when_price_stays_safe(
        self, sim, btc_usdt_perp: Symbol, side, safe_low, safe_high
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=side, quantity=1.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.entry_price)

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, low=safe_low, high=safe_high, close=50_000.0
        )

        assert len(sim.open_positions) == 1

    async def test_liquidation_cancels_pending_orders(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.0,
            stop_loss=StopLoss(trigger_price=45_000.0),
            take_profit=TakeProfit(trigger_price=55_000.0),
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.entry_price)

        assert len(sim.open_orders) == 2

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, low=39_000.0, high=42_000.0, close=41_000.0
        )

        assert len(sim.open_positions) == 0
        assert len(sim.open_orders) == 0

    @pytest.mark.parametrize(
        ("side", "liquidating_low", "liquidating_high"),
        [
            (OrderSide.BUY, 39_000.0, 42_000.0),
            (OrderSide.SELL, 58_000.0, 61_000.0),
        ],
    )
    async def test_liquidation_removes_margin_from_balance(
        self,
        sim,
        btc_usdt_perp: Symbol,
        side,
        liquidating_low,
        liquidating_high,
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=side, quantity=self.quantity
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.entry_price)
        balance_before = sim.simulation_engine.get_balances()["USDT"].total

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, low=liquidating_low, high=liquidating_high, close=50_000.0
        )

        balance_after = sim.simulation_engine.get_balances()["USDT"]
        assert balance_after == Balance(locked=0, total=balance_before - self.margin)

    @pytest.mark.parametrize(
        ("side", "expected_transaction_type", "expected_position_side"),
        [
            (OrderSide.BUY, "liquidate_long", PositionSide.LONG),
            (OrderSide.SELL, "liquidate_short", PositionSide.SHORT),
        ],
    )
    async def test_liquidation_records_transaction(
        self,
        sim,
        btc_usdt_perp: Symbol,
        side,
        expected_transaction_type,
        expected_position_side,
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=side, quantity=self.quantity
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=self.entry_price)
        position = sim.open_positions[btc_usdt_perp]
        liquidation_price = position.liquidation_price

        if side == OrderSide.BUY:
            sim.simulate_on_current_ohlcvs(
                btc_usdt_perp, low=39_000.0, high=42_000.0, close=41_000.0
            )
        else:
            sim.simulate_on_current_ohlcvs(
                btc_usdt_perp, low=58_000.0, high=61_000.0, close=59_000.0
            )

        transactions = sim.fill_recorder.get_fills()
        liquidation_tx = transactions[
            transactions["fill_type"].str.startswith("liquidate")
        ]

        assert len(liquidation_tx) == 1
        tx = liquidation_tx.iloc[0]
        assert tx["fill_type"] == expected_transaction_type
        assert tx["side"] == expected_position_side.value
        assert tx["gross_quantity"] == self.quantity
        assert tx["net_quantity"] == self.quantity
        assert tx["price"] == liquidation_price
        assert tx["reason"] == "liquidation"
