import logging

import pytest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import FeeRates
from robottraderslab.exceptions import ExchangeRecoverableError
from robottraderslab.exchanges import (
    FuturesExchangeProtocol,
    OrderSide,
    StopLoss,
    TakeProfit,
)


async def place_limit_sell_order(exchange: FuturesExchangeProtocol, symbol: Symbol):
    await exchange.place_limit_order(
        symbol=symbol, side=OrderSide.SELL, quantity=2.0, price=100.0
    )


async def place_trigger_limit_buy_order(
    exchange: FuturesExchangeProtocol, symbol: Symbol
):
    await exchange.place_limit_order(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=2.0,
        price=100.0,
        trigger_price=105.0,
    )


async def place_market_buy_order(exchange: FuturesExchangeProtocol, symbol: Symbol):
    await exchange.place_market_order(symbol=symbol, side=OrderSide.BUY, quantity=2.0)


async def place_market_sell_order(exchange: FuturesExchangeProtocol, symbol: Symbol):
    await exchange.place_market_order(symbol=symbol, side=OrderSide.SELL, quantity=2.0)


@pytest.fixture
def fee_rates() -> FeeRates:
    return FeeRates(maker=0.01, taker=0.02)


class TestAccounting:
    """Testing balances management and equity"""

    maker = 0.01
    taker = 0.02

    @pytest.mark.parametrize(
        "place_order",
        [
            place_trigger_limit_buy_order,
            place_market_buy_order,
            place_market_sell_order,
        ],
    )
    async def test_placing_order_doesnt_lock_funds(
        self, sim, btc_usdt_perp, place_order
    ):
        await place_order(sim.exchange, btc_usdt_perp)

        self._assert_usdt_balance(
            sim, expected_locked=0.0, expected_total=sim.initial_usdt
        )
        assert sim.simulation_engine.get_equity("USDT") == sim.initial_usdt

    @pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
    async def test_placing_limit_entry_locks_funds(self, sim, btc_usdt_perp, side):
        quantity = 2.0
        limit_price = 100.0
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=side,
            quantity=quantity,
            price=limit_price,
        )

        fee = 0.0
        margin_locked = quantity * limit_price
        total = sim.initial_usdt - fee
        self._assert_usdt_balance(
            sim, expected_locked=margin_locked, expected_total=total
        )
        assert sim.simulation_engine.get_equity("USDT") == sim.initial_usdt

    async def test_cancelling_a_limit_entry_releases_its_funds(
        self, sim, btc_usdt_perp
    ):
        placed = await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=2.0, price=100.0
        )

        await sim.exchange.cancel_order_by_id(btc_usdt_perp, placed.order_id)

        self._assert_usdt_balance(
            sim, expected_locked=0.0, expected_total=sim.initial_usdt
        )

    async def test_cancelling_a_symbol_releases_every_limit_entry(
        self, sim, btc_usdt_perp
    ):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=2.0, price=100.0
        )
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=2.0, price=100.0
        )

        await sim.exchange.cancel_orders_for_symbol(btc_usdt_perp)

        self._assert_usdt_balance(
            sim, expected_locked=0.0, expected_total=sim.initial_usdt
        )

    async def test_limit_buy_order_executed(self, sim, btc_usdt_perp):
        quantity = 2.0
        limit_price = 100.0
        close = 120.0
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=quantity,
            price=limit_price,
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=50.0, high=150.0, close=close)

        fee = quantity * limit_price * self.maker
        margin_locked = quantity * limit_price - fee
        total = sim.initial_usdt - fee
        unrealised_PnL = sim.open_positions[btc_usdt_perp].quantity * (
            close - limit_price
        )
        self._assert_usdt_balance(
            sim, expected_locked=margin_locked, expected_total=total
        )
        assert sim.simulation_engine.get_equity("USDT") == pytest.approx(
            total + unrealised_PnL
        )

    async def test_market_buy_order_executed(self, sim, btc_usdt_perp):
        quantity = 2.0
        close = 120.0
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=quantity
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=close)

        fee = quantity * close * self.taker
        margin_locked = quantity * close - fee
        total = sim.initial_usdt - fee
        unrealised_PnL = sim.open_positions[btc_usdt_perp].quantity * (close - close)
        self._assert_usdt_balance(
            sim, expected_locked=margin_locked, expected_total=total
        )
        assert sim.simulation_engine.get_equity("USDT") == pytest.approx(
            total + unrealised_PnL
        )

    async def test_limit_sell_order_executed(self, sim, btc_usdt_perp):
        position = await self._take_position(sim, btc_usdt_perp)
        quantity = position.quantity
        avg_price = position.average_entry_price
        usdt_balance_after_entry = sim.simulation_engine.get_balances()["USDT"]

        limit_price = 120.0
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=quantity,
            price=limit_price,
            reduce_only=True,
        )

        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp, open=limit_price, low=50.0, high=150.0, close=140.0
        )

        fee = quantity * limit_price * self.maker
        realised_PnL = quantity * (limit_price - avg_price)
        total = usdt_balance_after_entry.total + realised_PnL - fee
        self._assert_usdt_balance(sim, expected_locked=0.0, expected_total=total)
        assert sim.simulation_engine.get_equity("USDT") == pytest.approx(total)

    async def test_market_sell_order_executed(self, sim, btc_usdt_perp):
        position = await self._take_position(sim, btc_usdt_perp)
        quantity = position.quantity
        avg_price = position.average_entry_price
        usdt_balance_after_entry = sim.simulation_engine.get_balances()["USDT"]

        close = 120.0
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=50.0, high=150.0, close=close)
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.SELL,
            quantity=quantity,
            reduce_only=True,
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=close)

        fee = quantity * close * self.taker
        realised_PnL = quantity * (close - avg_price)
        total = usdt_balance_after_entry.total + realised_PnL - fee
        self._assert_usdt_balance(sim, expected_locked=0.0, expected_total=total)
        assert sim.simulation_engine.get_equity("USDT") == pytest.approx(total)

    async def test_not_enough_funds_to_place_order(self, sim, btc_usdt_perp):
        with pytest.raises(
            ExchangeRecoverableError, match="Insufficient available funds"
        ):
            await sim.exchange.place_limit_order(
                symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=2.0, price=100_000.0
            )

        self._assert_usdt_balance(
            sim, expected_locked=0.0, expected_total=sim.initial_usdt
        )
        assert sim.simulation_engine.get_equity("USDT") == sim.initial_usdt

    async def test_not_enough_funds_to_trigger_order(self, sim, btc_usdt_perp, caplog):
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            price=100_000.0,
            trigger_price=100.0,
        )

        with caplog.at_level(logging.WARNING):
            sim.simulate_on_current_ohlcvs(
                btc_usdt_perp, low=0.0, high=150.0, close=120.0
            )

        assert "Insufficient available funds" in caplog.text
        self._assert_usdt_balance(
            sim, expected_locked=0.0, expected_total=sim.initial_usdt
        )
        assert sim.simulation_engine.get_equity("USDT") == sim.initial_usdt

    async def test_not_enough_funds_to_execute_order(self, sim, btc_usdt_perp, caplog):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=2.0
        )

        with caplog.at_level(logging.WARNING):
            sim.simulate_on_current_ohlcvs(
                btc_usdt_perp, low=0.0, high=110_000.0, close=100_000.0
            )

        assert "Insufficient available funds" in caplog.text
        self._assert_usdt_balance(
            sim, expected_locked=0.0, expected_total=sim.initial_usdt
        )
        assert sim.simulation_engine.get_equity("USDT") == sim.initial_usdt

    async def test_not_enough_funds_to_execute_order_with_tpsl(
        self, sim, btc_usdt_perp, caplog
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=2.0,
            take_profit=TakeProfit(trigger_price=150_000.0),
            stop_loss=StopLoss(trigger_price=50_000.0),
        )

        with caplog.at_level(logging.WARNING):
            sim.simulate_on_current_ohlcvs(
                btc_usdt_perp, low=0.0, high=110_000.0, close=100_000.0
            )

        assert "Insufficient available funds" in caplog.text
        self._assert_usdt_balance(
            sim, expected_locked=0.0, expected_total=sim.initial_usdt
        )
        assert sim.simulation_engine.get_equity("USDT") == sim.initial_usdt
        assert len(sim.open_orders) == 0

    def _assert_usdt_balance(
        self, sim, *, expected_locked: float, expected_total: float
    ):
        usdt_balance = sim.simulation_engine.get_balances()["USDT"]
        assert usdt_balance.locked == pytest.approx(expected_locked), (
            f"Expected locked≈{expected_locked}; got {usdt_balance}"
        )
        assert usdt_balance.total == pytest.approx(expected_total), (
            f"Expected total≈{expected_total}; got {usdt_balance}"
        )
        assert usdt_balance.available == pytest.approx(expected_total - expected_locked)

    async def _take_position(self, sim, btc_usdt_perp: Symbol):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=2.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)
        return sim.open_positions[btc_usdt_perp]
