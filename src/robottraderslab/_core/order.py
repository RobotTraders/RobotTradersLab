import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol

from .price import validate_price
from .symbol import Symbol


class OrderType(StrEnum):
    """The kind of an order, read off an open order or a fill.

    A `TRIGGER` order waits for price to reach a level, then goes at market or
    at its limit. A `STOP_LOSS` or a `TAKE_PROFIT` protects a position and
    closes it when its level is reached. A `LIQUIDATION` is the venue closing
    a position whose margin ran out; no strategy places one.
    """

    LIMIT = "limit"
    MARKET = "market"
    TRIGGER = "trigger"
    STOP_LOSS = "stop-loss"
    TAKE_PROFIT = "take-profit"
    LIQUIDATION = "liquidation"


type ProtectionKind = Literal[OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]

type FillEffect = Literal["open", "increase", "reduce", "close"]
"""What a fill did to the position on its symbol."""

type FillSource = Literal["strategy", "take-profit", "stop-loss", "liquidation"]
"""What produced a fill: an order the strategy placed, an entry included, or
a protection or a liquidation the venue fired.
"""

VENUE_FIRED_SOURCES: dict[OrderType, FillSource] = {
    OrderType.STOP_LOSS: "stop-loss",
    OrderType.TAKE_PROFIT: "take-profit",
    OrderType.LIQUIDATION: "liquidation",
}

RESTING_ENTRY_KINDS: frozenset[OrderType] = frozenset(
    {OrderType.LIMIT, OrderType.TRIGGER}
)


class TimeInForce(StrEnum):
    """How long a limit order may wait on the book.

    A `GTC` order rests until it fills or is cancelled. An `IOC` order fills at
    once for what the book offers up to its price and the rest is cancelled. A
    `POST_ONLY` order is refused by the venue when it would fill at once, so it
    never takes liquidity.
    """

    GTC = "gtc"
    IOC = "ioc"
    POST_ONLY = "post_only"


class OrderSide(StrEnum):
    """The direction of an order: buy acquires the base currency, sell parts with it."""

    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> "OrderSide":
        return OrderSide.SELL if self == OrderSide.BUY else OrderSide.BUY


class OrderProtocol(Protocol):
    """An order resting on the venue, as the snapshot's open orders report it.

    Attributes:
        order_id: The id the venue assigned, which `cancel_order` and
            `modify_order` name the order by.
        kind: The kind of order, a protection included.
        trigger_price: The level a triggered order waits on, None on an
            order without a trigger.
        client_order_id: The id the order was placed with, which `tag_of`
            reads the tag from; None on a venue reporting none and on an
            order placed by hand.
    """

    order_id: str
    symbol: Symbol
    kind: OrderType
    side: OrderSide
    trigger_price: float | None
    client_order_id: str | None


class ProtectionVenue(Protocol):
    """The venue whose open orders a protection is confirmed against."""

    async def get_open_orders(self, symbols: Iterable[Symbol]) -> list[OrderProtocol]:
        """Return the resting orders on the symbols, triggers included.

        Args:
            symbols: Symbols the read is scoped to.
        """
        ...


@dataclass(kw_only=True, frozen=True)
class StopLoss:
    """Closes the position at market once price moves against it to the trigger.

    Attributes:
        reason: The strategy's own word for the exit, carried onto the fill.
    """

    trigger_price: float
    reason: str | None = None

    def __post_init__(self) -> None:
        """Refuse a trigger price no venue accepts.

        Raises:
            ValueError: If the trigger price is not positive or is NaN.
        """
        validate_price("trigger_price", self.trigger_price)


@dataclass(kw_only=True, frozen=True)
class TakeProfit:
    """Closes the position at market once price moves in its favour to the trigger.

    Attributes:
        reason: The strategy's own word for the exit, carried onto the fill.
    """

    trigger_price: float
    reason: str | None = None

    def __post_init__(self) -> None:
        """Refuse a trigger price no venue accepts.

        Raises:
            ValueError: If the trigger price is not positive or is NaN.
        """
        validate_price("trigger_price", self.trigger_price)


@dataclass(kw_only=True, frozen=True, slots=True)
class OrderPlacement:
    """An order the venue accepted, in the terms it was booked with.

    The price is absent for an order that goes at market, and the trigger price
    for one the venue holds no condition on. The client order id is absent on
    venues that take none, and the reason when the strategy booked the order
    without one.
    """

    order_id: str
    symbol: Symbol
    side: OrderSide
    quantity: float
    kind: OrderType
    price: float | None = None
    trigger_price: float | None = None
    client_order_id: str | None = None
    reason: str | None = None

    def __str__(self) -> str:
        payload: dict[str, object] = {
            "symbol": str(self.symbol),
            "side": self.side,
            "kind": self.kind,
            "qty": self.quantity,
            "order_id": self.order_id,
        }
        if self.price is not None:
            payload["price"] = self.price
        if self.trigger_price is not None:
            payload["trigger"] = self.trigger_price
        if self.reason is not None:
            payload["reason"] = self.reason
        return json.dumps(payload)


@dataclass(kw_only=True, frozen=True, slots=True)
class VenueFill:
    """A fill in the terms the venue reports it.

    The filled value is denominated in the symbol's quote currency and is
    absent on venues that report none. A venue reports what filled, never what
    the order was placed as, so no kind appears here.
    """

    order_id: str
    symbol: Symbol
    side: OrderSide
    quantity: float
    timestamp: datetime | None = None
    filled_value: float | None = None
    client_order_id: str | None = None


@dataclass(kw_only=True, frozen=True, slots=True)
class OrderFill:
    """A fill, in the terms of the order that produced it, handed to the
    order's `when_filled` callback.

    Attributes:
        quantity: What filled, in the base currency.
        kind: The kind of order that filled.
        timestamp: When the fill happened, None on a venue reporting no time.
        filled_value: What the fill was worth, in the symbol's quote currency,
            None on a venue reporting none.
        realised_profit: What the venue booked on the fill, in the symbol's
            margin currency, None where it reported none.
        client_order_id: The id the order was placed with, None on a venue
            taking none.
        reason: The strategy's own word for the order, None when it gave none.
        effect: None where the venue and the account state left it unsettled.
        source: None where it is unsettled.
    """

    order_id: str
    symbol: Symbol
    side: OrderSide
    quantity: float
    kind: OrderType
    timestamp: datetime | None = None
    filled_value: float | None = None
    realised_profit: float | None = None
    client_order_id: str | None = None
    reason: str | None = None
    effect: FillEffect | None = None
    source: FillSource | None = None

    def __str__(self) -> str:
        payload: dict[str, object] = {
            "time": str(self.timestamp),
            "symbol": str(self.symbol),
            "side": self.side,
            "kind": self.kind,
            "qty": self.quantity,
            "order_id": self.order_id,
        }
        if self.effect is not None:
            payload["effect"] = self.effect
        if self.source is not None:
            payload["source"] = self.source
        if self.filled_value is not None:
            payload["value"] = self.filled_value
        if self.realised_profit is not None:
            payload["profit"] = self.realised_profit
        if self.reason is not None:
            payload["reason"] = self.reason
        return json.dumps(payload)


@dataclass(kw_only=True, frozen=True, slots=True)
class Execution:
    """One execution the venue booked against the account, as the snapshot's
    fills report it.

    A venue reports an execution on its own, without pairing an exit to the
    entry it closes, so one execution may close one position and open
    another.

    Attributes:
        execution_id: Identifies this piece of a fill; one order can fill in
            several executions sharing its `order_id`.
        price: The price it filled at.
        quantity: What filled, in the base currency.
        timestamp: When it filled.
        kind: The kind of order that produced it, None on a venue that does
            not say.
        realised_profit: What the venue booked on this execution alone, in
            the symbol's margin currency; zero on one that only opened,
            None on a venue reporting none.
        fee: What the execution cost, in the symbol's margin currency, None
            on a venue reporting none.
        client_order_id: The id the order was placed with, None on an order
            placed outside the strategy.
        effect: What the execution did to the position, None until something
            settles it.
    """

    execution_id: str
    order_id: str
    symbol: Symbol
    side: OrderSide
    price: float
    quantity: float
    timestamp: datetime
    kind: OrderType | None = None
    realised_profit: float | None = None
    fee: float | None = None
    client_order_id: str | None = None
    effect: FillEffect | None = None


class FillDescriber(Protocol):
    """Gives the strategy's reason for a fill the venue produced on its own,
    called as `describe(fill: Execution) -> str | None`.

    A resting entry, limit or trigger, a stop-loss, a take-profit or a
    liquidation fills while no booked action is watching it, so the order's
    reason is out of reach when the fill is read back; the describer is where
    the strategy states it. The venue reports every fill on a declared
    symbol, so a limit or a trigger may belong to an order the strategy never
    placed: None on either disowns it and it goes unreported. A stop-loss,
    take-profit or liquidation acts on the position the strategy manages
    whatever placed it, so None there reports it by the kind of order that
    produced it.
    """

    def __call__(self, fill: Execution) -> str | None:
        """Return the reason for the fill, None when the strategy has none to give.

        Args:
            fill: A fill read back on a symbol the strategy declared, carrying
                the kind of order that produced it and the client order id the
                order was placed with.
        """
        ...


@dataclass(kw_only=True, frozen=True, slots=True)
class OrderRequest:
    """One order inside a batch placement.

    Attributes:
        limit_price: Limit price; a market order when None.
        time_in_force: A market order fills at once whatever it says.
        client_order_id: Caller-assigned order id, unchanged across retried
            attempts, so a venue that rejects a duplicate admits one order
            for the whole sequence.
    """

    symbol: Symbol
    side: OrderSide
    quantity: float
    limit_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    stop_loss: StopLoss | None = None
    take_profit: TakeProfit | None = None
    trigger_price: float | None = None
    client_order_id: str | None = None


@dataclass(kw_only=True, frozen=True, slots=True)
class OrderModifyRequest:
    """One resting order to reshape into a replacement order.

    Attributes:
        order_id: Exchange-assigned id of the resting order.
    """

    order_id: str
    order: OrderRequest


async def _order_not_filled() -> None:
    return None


type OnFillRead = Callable[["PlacedOrder", "OrderFill | None"], Awaitable[None]]
"""The callback `when_filled` registers: awaited once the order's fill is
read, in the cycle that placed the order, with the placed order and its fill,
None when the order did not fill in that cycle.
"""


@dataclass(kw_only=True, frozen=True, slots=True)
class PlacedOrder:
    """An order the venue accepted, and the means to read its fill.

    Asking for the fill while the order still rests reports None.

    Attributes:
        order_id: The id the venue assigned, which `cancel_order` names the
            order by.
    """

    order_id: str
    get_fill: Callable[[], Awaitable[VenueFill | None]] = _order_not_filled
