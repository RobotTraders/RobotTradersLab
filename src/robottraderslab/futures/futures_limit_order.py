from dataclasses import dataclass
from typing import Any, ClassVar

from robottraderslab._core import (
    ActionResult,
    BaseExchangeAction,
    OnFillRead,
    OrderOutcome,
    OrderPlacement,
    OrderSide,
    OrderType,
    StopLoss,
    Symbol,
    TakeProfit,
    TimeInForce,
)

from .futures_exchange_protocol import FuturesExchangeProtocol


@dataclass(kw_only=True, slots=True)
class FuturesLimitOrderAction(BaseExchangeAction):
    """Rests on the book, filling only at the given price or better, placed
    once the trigger price is reached when one is set.
    """

    kind: ClassVar[OrderType] = OrderType.LIMIT

    exchange: FuturesExchangeProtocol
    symbol: Symbol
    side: OrderSide
    price: float
    quantity: float
    reduce_only: bool = False
    stop_loss: StopLoss | None = None
    take_profit: TakeProfit | None = None
    trigger_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reason: str | None = None
    extra_fields: dict[str, Any] | None = None
    on_filled: OnFillRead | None = None

    async def execute(self) -> ActionResult:
        placed_order = await self.exchange.place_limit_order(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            price=self.price,
            reduce_only=self.reduce_only,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            trigger_price=self.trigger_price,
            client_order_id=self.client_order_id,
            time_in_force=self.time_in_force,
            reason=self.reason,
            extra_fields=self.extra_fields,
        )

        return ActionResult(
            orders=(
                OrderOutcome(
                    placed_order=placed_order,
                    on_filled=self.on_filled,
                    placement=OrderPlacement(
                        order_id=placed_order.order_id,
                        symbol=self.symbol,
                        side=self.side,
                        quantity=self.quantity,
                        kind=self.kind,
                        price=self.price,
                        trigger_price=self.trigger_price,
                        client_order_id=self.client_order_id,
                        reason=self.reason,
                    ),
                ),
            )
        )
