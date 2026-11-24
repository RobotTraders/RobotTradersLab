import pytest
from simulation_fixture import SimulationFixture

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FeeModel,
    FeeRates,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
    fee_model_for,
)
from robottraderslab.exchanges import (
    OrderSide,
    StopLoss,
)
from robottraderslab.futures import FuturesAccount
from robottraderslab_core.exchanges import PlacementReserve


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
