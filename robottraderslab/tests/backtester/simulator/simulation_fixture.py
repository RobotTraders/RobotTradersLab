from datetime import datetime
from typing import cast

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
)
from robottraderslab.exchanges import (
    OrderProtocol,
    OrderSide,
    PositionSnapshot,
    StopLoss,
    TakeProfit,
)
from robottraderslab_core.ohlcv import OHLCVRow, OHLCVsBySymbol


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
            if order.kind in ("stop-loss", "take-profit", "trailing-stop"):
                assert order.symbol in guarded_symbols, (
                    f"{order.kind} order resting for {order.symbol} with no open position"
                )


async def enter_position(
    sim: SimulationFixture,
    symbol: Symbol,
    side: OrderSide,
    *,
    quantity: float = 1.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> None:
    await sim.exchange.place_market_order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        stop_loss=StopLoss(trigger_price=stop_loss) if stop_loss is not None else None,
        take_profit=TakeProfit(trigger_price=take_profit)
        if take_profit is not None
        else None,
    )
    sim.simulate_on_current_ohlcvs(symbol, close=100.0)


def resting_guards(
    sim: SimulationFixture, kind: str
) -> list[tuple[float, float | None]]:
    return sorted(
        (order.trigger_price, order.quantity)
        for order in sim.open_orders
        if order.kind == kind
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
