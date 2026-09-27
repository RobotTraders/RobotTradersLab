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
from robottraderslab.exchanges import OrderProtocol, PositionSnapshot


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
            open=ohlcv.get("open", 0.0),
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
