import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from robottraderslab._internal import OrderSettings
from robottraderslab_core import Symbol
from robottraderslab_core.engine import ENGINE_TAGS, engine_tag_refusal, validate_price
from robottraderslab_core.exchanges import (
    NO_VENUE_PARAMS,
    OnFillRead,
    OrderSide,
    StopLoss,
    TakeProfit,
    TimeInForce,
    TriggerType,
    VenueParams,
)
from robottraderslab_core.futures import (
    BatchableOrderAction,
    FuturesLimitOrderAction,
    FuturesMarketOrderAction,
)
from robottraderslab_core.strategies import AccountSnapshot, ActionBuilder, SizingRule

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
        is_exit: Whether the order closes exposure the strategy holds. An exit
            is held to the quantity step alone, as a reduce-only order is.
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
    is_exit: bool = False
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
        is_exit: bool = False,
        reference_price: float | None = None,
    ) -> None:
        """Initialise the builder.

        Args:
            quantity: The quantity to trade, in the base currency; None leaves
                the sizing to `size`.
            settings: What the run's configuration sets on orders of the
                order's role, each winning over the chain's at `build()`.
            reduce_only: Whether the order may only reduce an existing position.
            is_exit: Whether the order closes exposure the strategy holds. An
                exit is held to the quantity step alone, as a reduce-only order
                is.
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
            is_exit=is_exit,
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
            ValueError: If the order is already sized, or if price is not
                positive or is NaN.
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

        Args:
            tag: The label written into the order's client order id.

        Raises:
            ValueError: If the tag is empty or blank, or is one the engine
                writes on its own orders: `sl`, `tp`, `sl-reached`,
                `tp-reached`, `flatten`.
        """
        validate_tag(tag)
        if tag in ENGINE_TAGS:
            raise engine_tag_refusal(tag)
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
            ValueError: If the order was never given a size, its rule
                cannot size it, or a triggered order was given a time in
                force other than `GTC`.
            BotCriticalError: If the symbol was never declared for the
                account.
        """
        quantity, denial = self._checked_quantity()
        limit_price = self._params.limit_price
        if limit_price is None:
            return self._market_order(quantity, denial)
        return self._limit_order(limit_price, quantity, denial)

    def _checked_quantity(self) -> tuple[float, str | None]:
        order_params = self._params
        return self._account._checked_quantity(
            order_params.symbol,
            self._sized_quantity(),
            price=order_params.limit_price
            or order_params.trigger_price
            or (
                order_params.reference_price
                if order_params.sizing is None
                else order_params.sizing.price
            ),
            reduce_only=order_params.reduce_only or order_params.is_exit,
        )

    def _limit_order(
        self, limit_price: float, quantity: float, denial: str | None
    ) -> FuturesLimitOrderAction:
        order_params = self._params
        overrides = self._account._order_overrides
        return FuturesLimitOrderAction(
            price=limit_price,
            exchange=self._account._exchange,
            symbol=order_params.symbol,
            side=order_params.side,
            quantity=quantity,
            reduce_only=order_params.reduce_only,
            is_exit=order_params.is_exit,
            stop_loss=overrides.stop_loss_over(order_params.stop_loss),
            take_profit=overrides.take_profit_over(order_params.take_profit),
            trigger_price=order_params.trigger_price,
            trigger_type=self._settings.trigger_type_over(order_params.trigger_type),
            time_in_force=self._time_in_force(),
            reason=order_params.reason,
            tag=order_params.tag,
            extra_fields=order_params.extra_fields,
            params=self._settings.params_over(order_params.venue_params),
            on_filled=order_params.on_filled,
            denial=denial,
        )

    def _market_order(
        self, quantity: float, denial: str | None
    ) -> FuturesMarketOrderAction:
        order_params = self._params
        overrides = self._account._order_overrides
        return FuturesMarketOrderAction(
            exchange=self._account._exchange,
            symbol=order_params.symbol,
            side=order_params.side,
            quantity=quantity,
            reduce_only=order_params.reduce_only,
            is_exit=order_params.is_exit,
            stop_loss=overrides.stop_loss_over(order_params.stop_loss),
            take_profit=overrides.take_profit_over(order_params.take_profit),
            trigger_price=order_params.trigger_price,
            trigger_type=self._settings.trigger_type_over(order_params.trigger_type),
            reason=order_params.reason,
            tag=order_params.tag,
            extra_fields=order_params.extra_fields,
            params=self._settings.params_over(order_params.venue_params),
            on_filled=order_params.on_filled,
            denial=denial,
        )

    def _sized_quantity(self) -> float:
        if self._params.sizing is not None:
            self._params.quantity = self._quantity_from_rule(self._params.sizing)
        if self._params.quantity is None:
            raise ValueError(
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
            raise ValueError(
                "The order is already sized; give it a quantity or a rule only once."
            )

    def _time_in_force(self) -> TimeInForce:
        order_params = self._params
        if order_params.trigger_price is None:
            return self._settings.time_in_force_over(order_params.time_in_force)
        if order_params.time_in_force is not TimeInForce.GTC:
            raise ValueError(
                f"A time in force of {order_params.time_in_force} on a triggered "
                "order: the trigger decides when the order goes, and it then rests "
                "until cancelled."
            )
        return order_params.time_in_force


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
