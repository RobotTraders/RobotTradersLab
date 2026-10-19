from typing import Any

from robottraderslab._core import (
    NO_ORDER_OVERRIDES,
    FeeMode,
    OrderOverrides,
    PlacementReserve,
)
from robottraderslab.bootstrap import SimulatedMarket
from robottraderslab.exchanges import Currency
from robottraderslab.futures import FuturesAccount

from .simulator import SimulatedFuturesExchange


def create_futures_market(
    *,
    initial_balance: dict[Currency, float],
    maker_fee_rate: float,
    taker_fee_rate: float,
    fee_mode: FeeMode,
    placement_reserve: PlacementReserve,
    order_overrides: OrderOverrides = NO_ORDER_OVERRIDES,
    **_kwargs: Any,
) -> SimulatedMarket:
    simulator = SimulatedFuturesExchange.create_from_settings(
        initial_balance=initial_balance,
        maker_fee_rate=maker_fee_rate,
        taker_fee_rate=taker_fee_rate,
        fee_mode=fee_mode,
        placement_reserve=placement_reserve,
    )
    return SimulatedMarket(
        simulator=simulator,
        account=FuturesAccount(
            simulator, name="simulated", order_overrides=order_overrides
        ),
    )
