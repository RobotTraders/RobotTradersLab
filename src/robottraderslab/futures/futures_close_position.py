import logging
from dataclasses import dataclass
from typing import Any

from robottraderslab._core import (
    ActionResult,
    BaseExchangeAction,
    OnFillRead,
    OrderOutcome,
    OrderPlacement,
    OrderSide,
    OrderType,
    PlacedOrder,
    Symbol,
    TimeInForce,
)

from .futures_exchange_protocol import FuturesExchangeProtocol

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class ClosePositionAction(BaseExchangeAction):
    """Closes the position the venue holds on the symbol, or a share of it,
    with a reduce-only order on the opposite side.

    Attributes:
        limit_price: Price the order rests at; unset, it goes at market.
    """

    exchange: FuturesExchangeProtocol
    symbol: Symbol
    closing_ratio: float = 1.0
    limit_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reason: str | None = None
    extra_fields: dict[str, Any] | None = None
    on_filled: OnFillRead | None = None

    async def execute(self) -> ActionResult:
        positions = await self.exchange.get_open_positions([self.symbol])
        position = positions.get(self.symbol)
        if position is None:
            logger.info("No position on `%s` to close", self.symbol)
            return ActionResult()
        side = position.side.closing_side
        quantity = position.quantity * self.closing_ratio
        placed_order = await self._place(side, quantity)
        return ActionResult(
            orders=(
                OrderOutcome(
                    placed_order=placed_order,
                    on_filled=self.on_filled,
                    placement=OrderPlacement(
                        order_id=placed_order.order_id,
                        symbol=self.symbol,
                        side=side,
                        quantity=quantity,
                        kind=OrderType.MARKET
                        if self.limit_price is None
                        else OrderType.LIMIT,
                        price=self.limit_price,
                        client_order_id=self.client_order_id,
                        reason=self.reason,
                    ),
                ),
            )
        )

    async def _place(self, side: OrderSide, quantity: float) -> PlacedOrder:
        if self.limit_price is None:
            return await self.exchange.place_market_order(
                symbol=self.symbol,
                side=side,
                quantity=quantity,
                reduce_only=True,
                client_order_id=self.client_order_id,
                reason=self.reason,
                extra_fields=self.extra_fields,
            )
        return await self.exchange.place_limit_order(
            symbol=self.symbol,
            side=side,
            quantity=quantity,
            price=self.limit_price,
            reduce_only=True,
            client_order_id=self.client_order_id,
            time_in_force=self.time_in_force,
            reason=self.reason,
            extra_fields=self.extra_fields,
        )


def validate_closing_ratio(closing_ratio: float) -> None:
    """A close takes a share of a position, never none of it.

    Raises:
        ValueError: If the ratio is outside (0, 1] or is NaN.
    """
    if not 0.0 < closing_ratio <= 1.0:
        raise ValueError(
            f"`closing_ratio` must be within (0, 1]; received {closing_ratio}"
        )
