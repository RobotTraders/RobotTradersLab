from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from robottraderslab_core import Symbol
from robottraderslab_core._internal import BaseExchangeAction
from robottraderslab_core._internal.actions.members import base_action
from robottraderslab_core.exchanges import (
    OrderProtocol,
    OrderSide,
    OrderType,
    TimeInForce,
    TriggerType,
)


@pytest.fixture(autouse=True)
def _restore_process_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base_action, "_order_id_prefix", base_action._order_id_prefix)
    monkeypatch.setattr(
        base_action, "_process_bot_bytes", base_action._process_bot_bytes
    )
    monkeypatch.setattr(
        base_action, "_process_bot_segment", base_action._process_bot_segment
    )


@dataclass(kw_only=True)
class RestingOrder:
    order_id: str = "order-7"
    symbol: Symbol = Symbol.create("BTC/USDT:USDT")
    kind: OrderType = OrderType.LIMIT
    side: OrderSide | None = OrderSide.SELL
    quantity: float | None = 2.0
    remaining_quantity: float = 2.0
    limit_price: float | None = 105.0
    trigger_price: float | None = None
    reduce_only: bool | None = True
    time_in_force: TimeInForce | None = TimeInForce.POST_ONLY
    trigger_type: TriggerType | None = None
    stop_loss_price: float | None = None
    stop_loss_trigger_type: TriggerType | None = None
    take_profit_price: float | None = None
    take_profit_trigger_type: TriggerType | None = None
    client_order_id: str | None = field(
        default_factory=lambda: (
            BaseExchangeAction(
                symbol=Symbol.create("BTC/USDT:USDT"), tag="1h-fast"
            ).client_order_id
        )
    )
    trailing_distance: float | None = None


@pytest.fixture
def create_resting_order() -> Callable[..., OrderProtocol]:
    return RestingOrder
