import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from robottraderslab._core import (
    NO_VENUE_PARAMS,
    AccountSnapshot,
    ActionBuilder,
    OnFillRead,
    OrderSettings,
    OrderSide,
    SizingRule,
    StopLoss,
    Symbol,
    TakeProfit,
    TimeInForce,
    TriggerType,
    VenueParams,
    validate_price,
)
from robottraderslab.exceptions import (
    ExchangeRecoverableError,
    StrategyCriticalError,
)

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
        reference_price: Price a market order's value is measured at against
            the venue's minimum order value; None for an order given its
            quantity with no price in view.
    """

    quantity: float | None
    side: OrderSide
    symbol: Symbol
    limit_price: float | None = None
    stop_loss: StopLoss | None = None
    take_profit: TakeProfit | None = None
    trigger_price: float | None = None
    trigger_type: TriggerType = TriggerType.DEFAULT
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    reason: str | None = None
    tag: str | None = None
    extra_fields: dict[str, Any] | None = None
    venue_params: VenueParams = NO_VENUE_PARAMS
    on_filled: OnFillRead | None = None
    sizing: _PendingSizing | None = None
    reference_price: float | None = None


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
        settings: OrderSettings,
        reduce_only: bool = False,
        reference_price: float | None = None,
    ) -> None:
        """Initialise the builder.

        Args:
            quantity: The quantity to trade, in the base currency; None leaves
                the sizing to `size`.
            settings: What the run's configuration sets on orders of the
                order's role, each winning over the chain's at `build()`.
            reduce_only: Whether the order may only reduce an existing position.
            reference_price: Price a market order's value is measured at
                against the venue's minimum order value; None for an order
                given its quantity with no price in view.

        Raises:
            ValueError: If the quantity is zero, negative or NaN, or the
                symbol carries no margin currency.
        """
        if quantity is not None and (quantity <= 0 or math.isnan(quantity)):
            raise ValueError(f"Quantity must be greater than 0; received {quantity}")
        _require_margin_currency(symbol)

        self._account = account
        self._settings = settings
        self._params = _FuturesOrderParams(
            quantity=quantity,
            side=side,
            symbol=symbol,
            reduce_only=reduce_only,
            reference_price=reference_price,
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
                `TimeInForce`; unless the order also waits on a trigger, a
                run's `[orders.entry]` or `[orders.exit]` `time_in_force`
                wins.

        Raises:
            ValueError: If price is not positive or is NaN.
        """
        validate_price("price", price)
        self._params.limit_price = price
        self._params.time_in_force = time_in_force
        return self

    def stop_loss(
        self,
        price: float,
        *,
        reason: str | None = None,
        trigger_type: TriggerType = TriggerType.DEFAULT,
    ) -> Self:
        """Attach a stop-loss the venue places together with the order.

        Args:
            reason: Label for the stop-loss (e.g. "SL 1", "tight stop"). An
                unnamed one is reported by the kind of order that fired it.
            trigger_type: The price the stop-loss watches; a type the venue
                offers no trigger on stops the run before the order is sent,
                and a run's `[orders.stop_loss]` `trigger_type` wins.

        Raises:
            ValueError: If price is not positive or is NaN.
        """
        self._params.stop_loss = StopLoss(
            trigger_price=price, reason=reason, trigger_type=trigger_type
        )
        return self

    def take_profit(
        self,
        price: float,
        *,
        reason: str | None = None,
        trigger_type: TriggerType = TriggerType.DEFAULT,
    ) -> Self:
        """Attach a take-profit the venue places together with the order.

        Args:
            reason: Label for the take-profit (e.g. "TP 1", "target 50pct").
                An unnamed one is reported by the kind of order that fired it.
            trigger_type: The price the take-profit watches; a type the venue
                offers no trigger on stops the run before the order is sent,
                and a run's `[orders.take_profit]` `trigger_type` wins.

        Raises:
            ValueError: If price is not positive or is NaN.
        """
        self._params.take_profit = TakeProfit(
            trigger_price=price, reason=reason, trigger_type=trigger_type
        )
        return self

    def trigger(
        self, price: float, *, trigger_type: TriggerType = TriggerType.DEFAULT
    ) -> Self:
        """Hold the order back until price reaches the level, then submit it
        at market, or at the limit when `.limit()` is also set.

        Args:
            trigger_type: The price the level watches; a type the venue
                offers no trigger on stops the run before the order is sent,
                and a run's `[orders.entry]` or `[orders.exit]`
                `trigger_type` wins.

        Raises:
            ValueError: If price is not positive or is NaN.
        """
        validate_price("price", price)
        self._params.trigger_price = price
        self._params.trigger_type = trigger_type
        return self

    def params(self, venue: str, **fields: Any) -> Self:
        """Send fields of one venue's own API in the request that places the
        order on that venue, named as its API names them; an order placed on
        another venue carries none of them.

        A field the connector sets from any order stops the run before the
        order is sent, even when this order leaves it unset, and a run's
        `[orders]` section wins over a field of the same name. A backtest
        ignores them.

        Args:
            venue: The name the venue's connector registers under, the one
                the account's configuration names, such as `bitget`.
            **fields: Merges into any fields an earlier call gave that venue.
        """
        self._params.venue_params = {
            **self._params.venue_params,
            venue: {**self._params.venue_params.get(venue, {}), **fields},
        }
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

        An order the venue's trading rules refuse is built carrying its
        denial, and nothing of it reaches the venue.

        Raises:
            StrategyCriticalError: If the order was never given a size, its
                rule cannot size it, a triggered order was given a time in
                force other than `GTC`, or the symbol was never declared for
                the account.
        """
        action: BatchableOrderAction

        order = self._params
        account = self._account
        symbol = order.symbol
        quantity = self._sized_quantity()
        denial = None
        try:
            quantity = (
                account._instruments.get(symbol) or account._instrument(symbol)
            ).order_quantity(
                symbol,
                quantity,
                price=order.limit_price
                or order.trigger_price
                or (
                    order.reference_price
                    if order.sizing is None
                    else order.sizing.price
                ),
                reduce_only=order.reduce_only,
            )
        except ExchangeRecoverableError as refusal:
            denial = str(refusal)
        exchange = account._exchange
        settings = self._settings
        time_in_force = order.time_in_force
        if order.trigger_price is None:
            time_in_force = settings.time_in_force_over(time_in_force)
        elif time_in_force is not TimeInForce.GTC:
            raise StrategyCriticalError(
                f"A time in force of {time_in_force} on a triggered order: the "
                "trigger decides when the order goes, and it then rests until "
                "cancelled."
            )
        params = settings.params_over(order.venue_params)
        trigger_type = settings.trigger_type_over(order.trigger_type)
        overrides = account._order_overrides
        stop_loss = overrides.stop_loss_over(order.stop_loss)
        take_profit = overrides.take_profit_over(order.take_profit)

        if order.limit_price is None:
            action = FuturesMarketOrderAction(
                exchange=exchange,
                symbol=order.symbol,
                side=order.side,
                quantity=quantity,
                reduce_only=order.reduce_only,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trigger_price=order.trigger_price,
                trigger_type=trigger_type,
                reason=order.reason,
                tag=order.tag,
                extra_fields=order.extra_fields,
                params=params,
                on_filled=order.on_filled,
                denial=denial,
            )
        else:
            action = FuturesLimitOrderAction(
                price=order.limit_price,
                exchange=exchange,
                symbol=order.symbol,
                side=order.side,
                quantity=quantity,
                reduce_only=order.reduce_only,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trigger_price=order.trigger_price,
                trigger_type=trigger_type,
                time_in_force=time_in_force,
                reason=order.reason,
                tag=order.tag,
                extra_fields=order.extra_fields,
                params=params,
                on_filled=order.on_filled,
                denial=denial,
            )

        return action

    def _sized_quantity(self) -> float:
        if self._params.sizing is not None:
            self._params.quantity = self._quantity_from_rule(self._params.sizing)
        if self._params.quantity is None:
            raise StrategyCriticalError(
                "The order has no size. Give it a quantity or size it with "
                "size(rule, price, account_snapshot) before it is built."
            )
        return self._params.quantity

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
