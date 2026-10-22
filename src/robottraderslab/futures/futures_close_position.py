import logging
from dataclasses import dataclass
from typing import Any

from robottraderslab._core import (
    NO_VENUE_PARAMS,
    ActionResult,
    BaseExchangeAction,
    Instrument,
    OnFillRead,
    OrderOutcome,
    OrderPlacement,
    OrderSide,
    OrderType,
    PlacedOrder,
    PositionSnapshot,
    Symbol,
    TimeInForce,
    VenueParams,
)

from .futures_exchange_protocol import FuturesExchangeProtocol

logger = logging.getLogger(__name__)

_WHOLE_POSITION = 1.0


@dataclass(kw_only=True, slots=True)
class ClosePositionAction(BaseExchangeAction):
    """Closes the position the venue holds on the symbol, or a share of it,
    with a reduce-only order on the opposite side.

    A share whose remainder would be worth less than the venue's minimum order
    value closes the whole position, so no close leaves a position too small
    to trade.

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
    params: VenueParams = NO_VENUE_PARAMS
    on_filled: OnFillRead | None = None

    async def execute(self) -> ActionResult:
        positions = await self.exchange.get_open_positions([self.symbol])
        position = positions.get(self.symbol)
        if position is None:
            logger.info("No position on `%s` to close", self.symbol)
            return ActionResult()
        side = position.side.closing_side
        quantity = await self._closing_quantity(position)
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

    async def _closing_quantity(self, position: PositionSnapshot) -> float:
        if self.closing_ratio == _WHOLE_POSITION:
            return position.quantity
        instrument = await self.exchange.get_instrument(self.symbol)
        share = instrument.order_quantity(
            self.symbol,
            position.quantity * self.closing_ratio,
            price=None,
            reduce_only=True,
        )
        if self._leaves_a_tradable_remainder(instrument, position, share):
            return share
        return position.quantity

    def _leaves_a_tradable_remainder(
        self, instrument: Instrument, position: PositionSnapshot, share: float
    ) -> bool:
        minimum = instrument.minimum_order_value
        if minimum is None:
            return True
        price = (
            position.average_entry_price
            if self.limit_price is None
            else self.limit_price
        )
        remainder_value = (position.quantity - share) * price
        if remainder_value >= minimum:
            return True
        logger.info(
            "%s: closing the whole position, since the remainder would be worth "
            "%g, under the minimum order value of %g",
            self.symbol,
            remainder_value,
            minimum,
        )
        return False

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
                params=self.params,
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
            params=self.params,
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
