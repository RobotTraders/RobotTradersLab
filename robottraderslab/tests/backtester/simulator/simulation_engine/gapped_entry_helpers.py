from datetime import datetime

from simulation_fixture import SimulationFixture

from robottraderslab import Symbol
from robottraderslab.exchanges import Execution, OrderSide, StopLoss, TakeProfit

MAKER = 0.001
TAKER = 0.005
QUANTITY = 2.0
LEVEL = 100.0


async def last_execution(sim: SimulationFixture) -> Execution:
    executions = await sim.exchange.get_executions_since(datetime.min, [])
    return executions[-1]


async def place_limit_entry(
    sim: SimulationFixture,
    symbol: Symbol,
    side: OrderSide,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> None:
    await sim.exchange.place_limit_order(
        symbol=symbol,
        side=side,
        quantity=QUANTITY,
        price=LEVEL,
        stop_loss=StopLoss(trigger_price=stop_loss) if stop_loss else None,
        take_profit=TakeProfit(trigger_price=take_profit) if take_profit else None,
    )


async def place_trigger_entry(
    sim: SimulationFixture,
    symbol: Symbol,
    side: OrderSide,
    placement_close: float,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    quantity: float = QUANTITY,
) -> None:
    sim.simulate_on_current_ohlcvs(symbol, close=placement_close)
    await sim.exchange.place_market_order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        trigger_price=LEVEL,
        stop_loss=StopLoss(trigger_price=stop_loss) if stop_loss else None,
        take_profit=TakeProfit(trigger_price=take_profit) if take_profit else None,
    )
