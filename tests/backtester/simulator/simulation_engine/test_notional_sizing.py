import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import OrderSide
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies import AccountSnapshot
from robottraderslab.strategies.futures import Notional

NOTIONAL = 200.0
PRICE = 100.0


async def _fill_notional(sim, symbol: Symbol, leverage: float) -> None:
    sim.simulation_engine.set_symbol_leverage(symbol, leverage)
    snapshot = AccountSnapshot(
        account_name="simulator",
        balances=sim.simulation_engine.get_balances(),
        conversion_rates={symbol: 1.0},
    )
    order = (
        FuturesAccount(sim.exchange)
        .long_entry(symbol)
        .size(Notional(NOTIONAL), PRICE, snapshot)
        .build()
    )
    await sim.exchange.place_market_order(
        symbol=symbol, side=OrderSide.BUY, quantity=order.quantity
    )
    sim.simulate_on_current_ohlcvs(symbol, close=PRICE)


class TestNotionalAtLeverage:
    @pytest.mark.parametrize("leverage", [1.0, 2.0, 10.0])
    async def test_the_position_is_worth_the_notional_less_the_reserve(
        self, sim, btc_usdt_perp, leverage
    ):
        await _fill_notional(sim, btc_usdt_perp, leverage)

        worth = sim.open_positions[btc_usdt_perp].quantity * PRICE
        assert worth == pytest.approx(
            NOTIONAL / (1 + sim.exchange.placement_reserve_rate)
        )

    @pytest.mark.parametrize("leverage", [1.0, 2.0, 10.0])
    async def test_the_leverage_sets_only_the_margin_locked(
        self, sim, btc_usdt_perp, leverage
    ):
        await _fill_notional(sim, btc_usdt_perp, leverage)

        worth = sim.open_positions[btc_usdt_perp].quantity * PRICE
        locked = sim.simulation_engine.get_balances()["USDT"].locked
        assert locked == pytest.approx(worth / leverage)
