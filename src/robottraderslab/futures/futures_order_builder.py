import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from robottraderslab._core import (
    AccountSnapshot,
    ActionBuilder,
    OnFillRead,
    OrderSide,
    SizingRule,
    StopLoss,
    Symbol,
    TakeProfit,
    TimeInForce,
    validate_price,
)
from robottraderslab.exceptions import StrategyCriticalError

from .futures_limit_order import FuturesLimitOrderAction
from .futures_market_order import FuturesMarketOrderAction
from .futures_order_batch import BatchableOrderAction

if TYPE_CHECKING:
    from .futures_account import FuturesAccount


@dataclass(frozen=True, slots=True)
class _PendingSizing:
    rule: SizingRule
    price: float
    account_snapshot: AccountSnapshot


@dataclass(kw_only=True)
class _FuturesOrderParams:
    """Internal storage for futures order parameters until build() is called.

    Attributes:
        quantity: The quantity to trade, in base currency.
        limit_price: If unset, a market order is created.
        reduce_only: Whether the order may only reduce an existing position.
        on_filled: Async callback invoked once the order's fill is read.
        sizing: The rule `build()` sizes the order with; None when the order
            is sized by a quantity.
    """

    quantity: float | None
    side: OrderSide
    symbol: Symbol
    limit_price: float | None = None
    stop_loss: StopLoss | None = None
    take_profit: TakeProfit | None = None
    trigger_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    reason: str | None = None
    tag: str | None = None
    extra_fields: dict[str, Any] | None = None
    on_filled: OnFillRead | None = None
    sizing: _PendingSizing | None = None


class FuturesOrderBuilder(ActionBuilder[BatchableOrderAction]):
    """An order being put together: an entry, an exit or the close of a
    tracked position, and a position target once it is sized.

    `limit` and `trigger` set the shape; without them the order goes at
    market. Every other setter adds an aspect the order carries whatever its
    shape.
    """

    def __init__(
        self,
        *,
        account: "FuturesAccount",
        quantity: float | None,
        side: OrderSide,
        symbol: Symbol,
        reduce_only: bool = False,
    ) -> None:
        """Initialise the builder.

        Args:
            quantity: The quantity to trade, in the base currency; None leaves
                the sizing to `size`.
            reduce_only: Whether the order may only reduce an existing position.

        Raises:
            ValueError: If the quantity is zero, negative or NaN, or the
                symbol carries no margin currency.
        """
        if quantity is not None and (quantity <= 0 or math.isnan(quantity)):
            raise ValueError(f"Quantity must be greater than 0; received {quantity}")
        _require_margin_currency(symbol)

        self._account = account
        self._params = _FuturesOrderParams(
            quantity=quantity,
            side=side,
            symbol=symbol,
            reduce_only=reduce_only,
        )

    def size(
        self, rule: SizingRule, price: float, account_snapshot: AccountSnapshot
    ) -> Self:
        """Size the order with a rule, typically the one the configuration
        names for the profile.

        The rule sizes the order when it is built, so it reads the stop-loss
        attached anywhere on the chain.

        Args:
            price: Price the order is expected to fill at, which the rule
                converts its amount into a quantity at.
            account_snapshot: The account's snapshot for this candle; passing
                the rule to `requirements.account.add` as `sizing` declares
                the state it reads.

        Raises:
            StrategyCriticalError: If the order is already sized.
            ValueError: If price is not positive or is NaN.
        """
        self._require_unsized()
        validate_price("price", price)
        self._params.sizing = _PendingSizing(rule, price, account_snapshot)
        return self

    def limit(
        self, price: float, *, time_in_force: TimeInForce = TimeInForce.GTC
    ) -> Self:
        """Rest the order at a limit price; without it the order goes at market.

        Args:
            time_in_force: How long the order may wait on the book, one of
                `TimeInForce`.

        Raises:
            ValueError: If price is not positive or is NaN.
        """
        validate_price("price", price)
        self._params.limit_price = price
        self._params.time_in_force = time_in_force
        return self

    def stop_loss(self, price: float, *, reason: str | None = None) -> Self:
        """Attach a stop-loss the venue places together with the order.

        Args:
            reason: Label for the stop-loss (e.g. "SL 1", "tight stop"). An
                unnamed one is reported by the kind of order that fired it.

        Raises:
            ValueError: If price is not positive or is NaN.
        """
        self._params.stop_loss = StopLoss(trigger_price=price, reason=reason)
        return self

    def take_profit(self, price: float, *, reason: str | None = None) -> Self:
        """Attach a take-profit the venue places together with the order.

        Args:
            reason: Label for the take-profit (e.g. "TP 1", "target 50pct").
                An unnamed one is reported by the kind of order that fired it.

        Raises:
            ValueError: If price is not positive or is NaN.
        """
        self._params.take_profit = TakeProfit(trigger_price=price, reason=reason)
        return self

    def trigger(self, price: float) -> Self:
        """Hold the order back until price reaches the level, then submit it
        at market, or at the limit when `.limit()` is also set.

        Raises:
            ValueError: If price is not positive or is NaN.
        """
        validate_price("price", price)
        self._params.trigger_price = price
        return self

    def reason(self, reason: str) -> Self:
        """Explain why the order was placed, for later analysis.

        Args:
            reason: What triggered the order (e.g. "MA crossover",
                "z score -2", "overbought").
        """
        self._params.reason = reason
        return self

    def tag(self, tag: str) -> Self:
        """Label the order so a later run recognises it as its own.

        The label travels in the client order id the venue echoes back on the
        order, where `tag_of` reads it; a profile's `order_tag` fits every
        venue's client order id.

        Raises:
            ValueError: If the tag is empty.
        """
        validate_tag(tag)
        self._params.tag = tag
        return self

    def extra_fields(self, **fields: Any) -> Self:
        """Record custom fields on the backtest's fill; a venue receives none.

        Args:
            **fields: Merges into any fields an earlier call gave.
        """
        if self._params.extra_fields is None:
            self._params.extra_fields = {}
        self._params.extra_fields.update(fields)
        return self

    def when_filled(self, callback: OnFillRead) -> Self:
        """Register a callback invoked once the order's fill is read.

        Args:
            callback: Receives the placed order and its fill,
                None if it did not fill.
        """
        self._params.on_filled = callback
        return self

    def build(self) -> BatchableOrderAction:
        """Return the finished action. The bookkeeper and the batch methods
        finish a builder handed to them with this call, at the moment it is
        handed over.

        Raises:
            StrategyCriticalError: If the order was never given a size, its
                rule cannot size it, or a triggered order was given a time in
                force other than `GTC`.
        """
        action: BatchableOrderAction

        if self._params.sizing is not None:
            self._params.quantity = self._quantity_from_rule(self._params.sizing)
        if self._params.quantity is None:
            raise StrategyCriticalError(
                "The order has no size. Give it a quantity or size it with "
                "size(rule, price, account_snapshot) before it is built."
            )
        if (
            self._params.trigger_price is not None
            and self._params.time_in_force is not TimeInForce.GTC
        ):
            raise StrategyCriticalError(
                f"A time in force of {self._params.time_in_force} on a triggered "
                "order: the trigger decides when the order goes, and it then "
                "rests until cancelled."
            )

        exchange = self._account._exchange

        if self._params.limit_price is None:
            action = FuturesMarketOrderAction(
                exchange=exchange,
                symbol=self._params.symbol,
                side=self._params.side,
                quantity=self._params.quantity,
                reduce_only=self._params.reduce_only,
                stop_loss=self._params.stop_loss,
                take_profit=self._params.take_profit,
                trigger_price=self._params.trigger_price,
                reason=self._params.reason,
                tag=self._params.tag,
                extra_fields=self._params.extra_fields,
                on_filled=self._params.on_filled,
            )
        else:
            action = FuturesLimitOrderAction(
                price=self._params.limit_price,
                exchange=exchange,
                symbol=self._params.symbol,
                side=self._params.side,
                quantity=self._params.quantity,
                reduce_only=self._params.reduce_only,
                stop_loss=self._params.stop_loss,
                take_profit=self._params.take_profit,
                trigger_price=self._params.trigger_price,
                time_in_force=self._params.time_in_force,
                reason=self._params.reason,
                tag=self._params.tag,
                extra_fields=self._params.extra_fields,
                on_filled=self._params.on_filled,
            )

        return action

    def _quantity_from_rule(self, sizing: _PendingSizing) -> float:
        stop_loss = self._params.stop_loss
        return sizing.rule.quantity(
            self._params.symbol,
            sizing.price,
            sizing.account_snapshot,
            placement_reserve_rate=self._account.placement_reserve_rate,
            placement_requirement_rate=self._account.placement_requirement_rate,
            stop_loss_price=None if stop_loss is None else stop_loss.trigger_price,
        )

    def _require_unsized(self) -> None:
        if self._params.quantity is not None or self._params.sizing is not None:
            raise StrategyCriticalError(
                "The order is already sized; give it a quantity or a rule only once."
            )


def validate_tag(tag: str) -> None:
    """A tag is carried in a client order id, where `tag_of` reads nothing
    back from a blank one.

    Raises:
        ValueError: If the tag is empty or blank.
    """
    if not tag.strip():
        raise ValueError("Tag cannot be empty")


def _require_margin_currency(symbol: Symbol) -> None:
    if symbol.margin is None:
        raise ValueError(
            f"FuturesOrderBuilder requires a futures symbol with a margin currency; got '{symbol}'"
        )
