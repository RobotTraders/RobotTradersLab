from datetime import datetime
from typing import cast

import pytest

from robottraderslab import Symbol
from robottraderslab._core import OHLCVRow, OHLCVsBySymbol, PlacementReserve
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FeeModel,
    FeeRates,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
    fee_model_for,
)
from robottraderslab.exchanges import (
    OrderProtocol,
    OrderSide,
    PositionSnapshot,
    StopLoss,
)
from robottraderslab.futures import FuturesAccount


class SimulationFixture:
    """Provides a FuturesSimulationEngine + SimulatedFuturesExchange pair for testing."""

    initial_usdt = 10_000.0

    def __init__(
        self,
        simulation_engine: FuturesSimulationEngine,
        exchange: SimulatedFuturesExchange,
        fill_recorder: CacheFillRecorder,
    ):
        self.simulation_engine = simulation_engine
        self.exchange = exchange
        self.fill_recorder = fill_recorder

    @property
    def open_orders(self) -> list[OrderProtocol]:
        return cast(list[OrderProtocol], self.simulation_engine.open_orders)

    @property
    def open_positions(self) -> dict[Symbol, PositionSnapshot]:
        return self.simulation_engine.open_positions

    def simulate_on_current_ohlcvs(self, symbol: Symbol, **ohlcv: float) -> None:
        ohlcvs_by_symbol = _create_ohlcvs_by_symbol(symbol, **ohlcv)
        self.simulation_engine.simulate_on_current_ohlcvs(
            datetime.now(), ohlcvs_by_symbol
        )
        self._assert_no_orphan_guards()

    def _assert_no_orphan_guards(self) -> None:
        guarded_symbols = set(self.open_positions)
        for order in self.open_orders:
            if order.kind in ("stop-loss", "take-profit"):
                assert order.symbol in guarded_symbols, (
                    f"{order.kind} order resting for {order.symbol} with no open position"
                )


def _create_ohlcvs_by_symbol(symbol: Symbol, **ohlcv: float) -> OHLCVsBySymbol:
    if "close" in ohlcv:
        ohlcv.setdefault("low", ohlcv["close"])
        ohlcv.setdefault("high", ohlcv["close"])
    return {
        symbol: OHLCVRow(
            timestamp=datetime.now(),
            open=ohlcv.get("open", ohlcv.get("close", 0.0)),
            high=ohlcv.get("high", 0.0),
            low=ohlcv.get("low", 0.0),
            close=ohlcv.get("close", 0.0),
            volume=ohlcv.get("volume", 0.0),
        )
    }


@pytest.fixture(scope="session")
def btc_usdt_perp() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


@pytest.fixture(scope="session")
def eth_usdt_perp() -> Symbol:
    return Symbol.create("ETH/USDT:USDT")


@pytest.fixture
def fee_rates() -> FeeRates:
    return FeeRates(maker=0.0, taker=0.0)


@pytest.fixture
def fee_model() -> FeeModel:
    return fee_model_for("cost", PlacementReserve())


@pytest.fixture
def sim_without_recording(
    fee_rates: FeeRates, fee_model: FeeModel
) -> SimulationFixture:
    """A simulation fixture with execution recording left at its default, for
    tests of the recording gate itself.
    """
    margin_currency = "USDT"
    fill_recorder = CacheFillRecorder()
    simulation_engine = FuturesSimulationEngine(
        initial_balance={margin_currency: SimulationFixture.initial_usdt},
        fee_rates=fee_rates,
        fill_recorder=fill_recorder,
        margin_currency=margin_currency,
        fee_model=fee_model,
    )
    exchange = SimulatedFuturesExchange(simulation_engine, fill_recorder)
    return SimulationFixture(simulation_engine, exchange, fill_recorder)


@pytest.fixture
def sim(sim_without_recording: SimulationFixture) -> SimulationFixture:
    sim_without_recording.simulation_engine.enable_execution_recording()
    return sim_without_recording


@pytest.fixture
async def declared_account(sim: SimulationFixture, btc_usdt_perp) -> FuturesAccount:
    """An account holding the simulator's trading rules for the symbol, which
    the simulator reports once it has a price for it.
    """
    sim.simulation_engine.record_prices({btc_usdt_perp: 100.0})
    account = FuturesAccount(sim.exchange)
    await account._read_instruments([btc_usdt_perp])
    return account


@pytest.fixture
async def holding_sim(
    sim: SimulationFixture, btc_usdt_perp: Symbol, eth_usdt_perp: Symbol
) -> tuple[SimulationFixture, tuple[Symbol, ...]]:
    """A simulator holding a BTC long reopened after its stop-loss fired, a resting
    ETH buy, and a EUR-quoted symbol with its rate recorded.
    """
    euro_perp = Symbol.create("ETH/EUR:USDT")
    sim.simulation_engine.record_prices({Symbol("EUR", "USDT"): 1.1})
    await sim.exchange.place_market_order(
        symbol=btc_usdt_perp,
        side=OrderSide.BUY,
        quantity=1.0,
        stop_loss=StopLoss(trigger_price=150.0),
    )
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=180.0, high=220.0, close=200.0)
    await sim.exchange.place_limit_order(
        symbol=eth_usdt_perp, side=OrderSide.BUY, quantity=2.0, price=50.0
    )
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, low=140.0, high=200.0, close=160.0)
    await sim.exchange.place_market_order(
        symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=1.0
    )
    return sim, (btc_usdt_perp, eth_usdt_perp, euro_perp)
