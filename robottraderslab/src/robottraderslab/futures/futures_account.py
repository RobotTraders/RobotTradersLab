import logging
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime

from robottraderslab._internal import (
    NO_ORDER_OVERRIDES,
    NO_ORDER_SETTINGS,
    OrderOverrides,
    OrderSettings,
)
from robottraderslab.exceptions import BotCriticalError
from robottraderslab_core import Symbol
from robottraderslab_core.engine import (
    BASE_DELAY_SECONDS,
    ENGINE_TAGS,
    FLATTEN_TAG,
    MAX_ATTEMPTS,
    AccountRequirement,
    Halt,
    OrderReplacement,
    attribute_fill_effects,
    finished,
    validate_closing_ratio,
    validate_price,
    validate_ratio,
)
from robottraderslab_core.exceptions import SymbolNotListedError
from robottraderslab_core.exchanges import (
    AccountStateRequest,
    BaseExchangeAction,
    Execution,
    Instrument,
    MarginMode,
    OrderProtocol,
    OrderRequest,
    OrderSide,
    PositionSide,
    TimeInForce,
    TriggerType,
    modify_request,
    retry_on_transient,
    tag_of,
)
from robottraderslab_core.futures import (
    BatchableOrder,
    CancelOrderByIdAction,
    CancelOrdersForSymbolAction,
    ClosePositionAction,
    FuturesCapabilities,
    FuturesExchangeProtocol,
    FuturesOrderBatchAction,
    FuturesOrderModifyAction,
    OrderModification,
    SetLeverageAction,
    SetMarginModeAction,
    UpdatePositionStopLossAction,
    UpdatePositionTakeProfitAction,
    UpdatePositionTrailingStopAction,
)
from robottraderslab_core.strategies import AccountSnapshot, TrackedPosition

from .futures_close_builder import FuturesCloseBuilder
from .futures_order_builder import FuturesOrderBuilder, validate_tag
from .futures_position_target import MINIMUM_ORDER_RATIO, FuturesPositionTarget
from .futures_trailing_stop_loss import validate_trailing_ratio

logger = logging.getLogger(__name__)


class FuturesAccount:
    """A strategy's trading account, and the factory for its actions.

    Each method returns what the strategy books. An entry, an exit or a
    tracked close returns an order builder. A symbol's close returns a close
    builder. A target returns a `FuturesPositionTarget`, sized into an order
    builder. Every other method returns a finished action.
    """

    def __init__(
        self,
        exchange: FuturesExchangeProtocol,
        name: str | None = None,
        order_overrides: OrderOverrides = NO_ORDER_OVERRIDES,
        venue: str | None = None,
    ):
        """Initialise the account.

        Args:
            exchange: Venue the account trades on.
            name: Identifies the account among the ones a strategy uses, and
                labels it in errors and logs. Defaults to the exchange type.
            order_overrides: The order settings the run's configuration
                sets; each takes the place of the strategy's setting on every
                order the account builds.
            venue: The exchange the account trades on, as the configuration
                names it, such as `bitget`, which labels the account's venue
                in logs. Defaults to the exchange type.
        """
        self._exchange = exchange
        self.name = name or type(exchange).__name__
        self._venue = venue or type(exchange).__name__
        self._order_overrides = order_overrides
        self._instruments: dict[Symbol, Instrument] = {}
        self._unlisted: dict[Symbol, str] = {}

    @property
    def capabilities(self) -> FuturesCapabilities:
        """Behavioural traits declared by the underlying exchange."""
        return self._exchange.capabilities

    @property
    def placement_reserve_rate(self) -> float:
        """Share of an order's value the venue holds back at placement beyond
        the margin, handed to every sizing rule.
        """
        return self._exchange.placement_reserve_rate

    def cancel_order(self, symbol: Symbol, order_id: str) -> CancelOrderByIdAction:
        """Cancel a single open order by its exchange-assigned id."""
        return CancelOrderByIdAction(
            exchange=self._exchange,
            symbol=symbol,
            order_id=order_id,
        )

    def cancel_orders(
        self, symbol: Symbol, tag: str | None = None
    ) -> CancelOrdersForSymbolAction:
        """Cancel the open orders on a symbol, protections included, or only
        the ones carrying `tag` when it is given.

        The orders are read from the venue when the action runs, so an order
        booked earlier in the same cycle is among them.

        Args:
            tag: The tag `tag_of` reads from an order's client order id; every
                order without it, the position's protections included, stays
                on the book.

        Raises:
            ValueError: If the tag is empty.
        """
        if tag is not None:
            validate_tag(tag)
        return CancelOrdersForSymbolAction(
            exchange=self._exchange,
            symbol=symbol,
            order_tag=tag,
        )

    def close_position(
        self, symbol: Symbol, closing_ratio: float = 1.0
    ) -> FuturesCloseBuilder:
        """Start the order that closes the position the venue holds on the
        symbol, or a share of it.

        The position is read when the action runs, so the order closes what
        is held then, reduce-only on the opposite side; a symbol found flat
        places nothing. A profile stacked with others on the symbol closes
        its own share with `close_tracked_position`, since the venue reports
        one netted position.

        Args:
            closing_ratio: Share of the position the order closes, the whole
                of it by default.

        Raises:
            ValueError: If the ratio is not within (0, 1].
        """
        validate_closing_ratio(closing_ratio)
        return FuturesCloseBuilder(
            exchange=self._exchange,
            symbol=symbol,
            closing_ratio=closing_ratio,
            settings=self._order_overrides.exit,
        )

    def close_tracked_position(
        self, symbol: Symbol, tracked: TrackedPosition, closing_ratio: float = 1.0
    ) -> FuturesOrderBuilder:
        """Start the exit that closes the position a tracker holds under one
        id, or a share of it.

        The order takes the side opposite the tracked position and is not
        reduce-only, since the venue nets every profile on the symbol into
        one position the tracked share may stand against.

        Args:
            tracked: The position a `PositionTracker` holds under the id.
            closing_ratio: Share of the tracked position the order closes, the
                whole of it by default.

        Returns:
            A builder already sized from the tracked position.

        Raises:
            ValueError: If the ratio is not within (0, 1] or the symbol
                carries no margin currency.
        """
        validate_closing_ratio(closing_ratio)
        quantity = tracked.quantity * closing_ratio
        if tracked.side is PositionSide.LONG:
            return self._create_sell_order(
                symbol, quantity, self._order_overrides.exit, reduce_only=False
            )
        return self._create_buy_order(
            symbol, quantity, self._order_overrides.exit, reduce_only=False
        )

    def long_entry(
        self, symbol: Symbol, quantity: float | None = None
    ) -> FuturesOrderBuilder:
        """Start a buy order that opens or grows a long position.

        Args:
            quantity: Quantity to buy, in the base currency; omit it to size
                the order with the builder's `size`.

        Raises:
            ValueError: If the quantity is zero, negative or NaN, or the
                symbol carries no margin currency.
        """
        return self._create_buy_order(symbol, quantity, self._order_overrides.entry)

    def long_exit(
        self,
        symbol: Symbol,
        quantity: float | None = None,
        *,
        reduce_only: bool = True,
    ) -> FuturesOrderBuilder:
        """Start a sell order that closes a long position.

        Args:
            quantity: Quantity to sell, in the base currency; omit it to size
                the order with the builder's `size`.
            reduce_only: Whether the order may only shrink the position,
                never open the opposite side.

        Raises:
            ValueError: If the quantity is zero, negative or NaN, or the
                symbol carries no margin currency.
        """
        return self._create_sell_order(
            symbol, quantity, self._order_overrides.exit, reduce_only=reduce_only
        )

    def long_target(
        self,
        symbol: Symbol,
        account_snapshot: AccountSnapshot,
        *,
        minimum_order_ratio: float = MINIMUM_ORDER_RATIO,
    ) -> FuturesPositionTarget:
        """State where a long position on the symbol should stand.

        Args:
            account_snapshot: The account's snapshot for this candle, read
                with `positions=True` and `balances=True`, and equity when the
                target is a share of it.
            minimum_order_ratio: Share of the total balance under which the
                difference is left alone, a tenth of a percent by default.

        Raises:
            ValueError: If the minimum is outside [0, 1].
        """
        validate_ratio("minimum_order_ratio", minimum_order_ratio)
        return FuturesPositionTarget(
            account=self,
            symbol=symbol,
            side=PositionSide.LONG,
            account_snapshot=account_snapshot,
            minimum_order_ratio=minimum_order_ratio,
        )

    def modify_order(
        self, order_id: str, order: BatchableOrder
    ) -> FuturesOrderModifyAction:
        """Reshape one resting order into another, keeping its reduce-only
        whatever the replacement carries.

        Args:
            order_id: Exchange-assigned id of the resting order.
            order: The replacement, a built order or an order builder the
                account finishes with its `build()`.

        Raises:
            ValueError: If the builder's `build()` refuses the order.
            BotCriticalError: If the order's symbol was never declared for the
                account.
        """
        return self.modify_orders([OrderModification(order_id=order_id, order=order)])

    def modify_orders(
        self, modifications: Iterable[OrderModification]
    ) -> FuturesOrderModifyAction:
        """Reshape several resting orders together.

        The venue receives them as few requests as it allows, and one
        rejected modification does not stop the others. A resting order
        that filled or was cancelled in the meantime is skipped, never
        re-placed, and each keeps its reduce-only whatever its replacement
        carries.

        Args:
            modifications: Resting orders and their replacements.

        Raises:
            ValueError: If no modifications are given, or a builder's
                `build()` refuses its order.
            BotCriticalError: If an order's symbol was never declared for the
                account.
        """
        replacements = [
            OrderReplacement(
                order_id=modification.order_id, order=finished(modification.order)
            )
            for modification in modifications
        ]
        if not replacements:
            raise ValueError("An order modification batch needs at least one order")
        return FuturesOrderModifyAction(
            exchange=self._exchange,
            symbol=replacements[0].order.symbol,
            modifications=replacements,
        )

    def move_stop_loss(
        self,
        symbol: Symbol,
        price: float,
        *,
        trigger_type: TriggerType = TriggerType.DEFAULT,
    ) -> UpdatePositionStopLossAction:
        """Move the stop-loss guarding an open position to a price.

        A position holding no stop-loss is given one. A move the venue rejects
        is sent once more in the same cycle, and one it still refuses is logged
        at ERROR; a move on a symbol the account holds no position on is
        dropped at INFO.

        Args:
            trigger_type: The price the stop-loss watches; a run's
                `[orders.stop_loss]` `trigger_type` wins, and a type the venue
                offers no trigger on stops the run.

        Raises:
            ValueError: If the price is not positive or is NaN.
        """
        validate_price("price", price)
        return UpdatePositionStopLossAction(
            exchange=self._exchange,
            symbol=symbol,
            trigger_price=price,
            trigger_type=self._order_overrides.stop_loss_trigger_type_over(
                trigger_type
            ),
        )

    def move_take_profit(
        self,
        symbol: Symbol,
        price: float,
        *,
        trigger_type: TriggerType = TriggerType.DEFAULT,
    ) -> UpdatePositionTakeProfitAction:
        """Move the take-profit on an open position to a price.

        A move the venue rejects is sent once more in the same cycle, and one
        it still refuses is logged at ERROR; a move on a symbol the account
        holds no position on is dropped at INFO.

        Args:
            trigger_type: The price the take-profit watches; a run's
                `[orders.take_profit]` `trigger_type` wins, and a type the venue
                offers no trigger on stops the run.

        Raises:
            ValueError: If the price is not positive or is NaN.
        """
        validate_price("price", price)
        return UpdatePositionTakeProfitAction(
            exchange=self._exchange,
            symbol=symbol,
            trigger_price=price,
            trigger_type=self._order_overrides.take_profit_trigger_type_over(
                trigger_type
            ),
        )

    def place_orders(self, orders: Iterable[BatchableOrder]) -> FuturesOrderBatchAction:
        """Place several orders together.

        The venue receives them as few requests as it allows, and one
        rejected order does not stop the others.

        Args:
            orders: Built orders, or order builders the account finishes with
                their `build()`.

        Raises:
            ValueError: If no orders are given, or a builder's `build()`
                refuses its order.
            BotCriticalError: If an order's symbol was never declared for the
                account.
        """
        built = [finished(order) for order in orders]
        if not built:
            raise ValueError("An order batch needs at least one order")
        return FuturesOrderBatchAction(
            exchange=self._exchange,
            symbol=built[0].symbol,
            orders=built,
        )

    def placement_requirement_rate(self, leverage: float) -> float:
        """Share of an order's value the venue locks at placement at a
        leverage, the margin included, handed to every sizing rule.
        """
        return self._exchange.placement_requirement_rate(leverage)

    def set_leverage(self, symbol: Symbol, leverage: float) -> SetLeverageAction:
        """Set the account's leverage on the symbol, from this candle's
        actions on; the leverage a symbol trades at throughout is its
        `margin_targets` entry.

        Raises:
            ValueError: If the leverage is below 1 or NaN.
        """
        return SetLeverageAction(
            exchange=self._exchange,
            symbol=symbol,
            leverage=leverage,
        )

    def set_margin_mode(
        self, symbol: Symbol, margin_mode: MarginMode
    ) -> SetMarginModeAction:
        """Set the account's margin mode on the symbol."""
        return SetMarginModeAction(
            exchange=self._exchange,
            symbol=symbol,
            margin_mode=margin_mode,
        )

    def short_entry(
        self, symbol: Symbol, quantity: float | None = None
    ) -> FuturesOrderBuilder:
        """Start a sell order that opens or grows a short position.

        Args:
            quantity: Quantity to sell, in the base currency; omit it to size
                the order with the builder's `size`.

        Raises:
            ValueError: If the quantity is zero, negative or NaN, or the
                symbol carries no margin currency.
        """
        return self._create_sell_order(symbol, quantity, self._order_overrides.entry)

    def short_exit(
        self,
        symbol: Symbol,
        quantity: float | None = None,
        *,
        reduce_only: bool = True,
    ) -> FuturesOrderBuilder:
        """Start a buy order that closes a short position.

        Args:
            quantity: Quantity to buy, in the base currency; omit it to size
                the order with the builder's `size`.
            reduce_only: Whether the order may only shrink the position,
                never open the opposite side.

        Raises:
            ValueError: If the quantity is zero, negative or NaN, or the
                symbol carries no margin currency.
        """
        return self._create_buy_order(
            symbol, quantity, self._order_overrides.exit, reduce_only=reduce_only
        )

    def short_target(
        self,
        symbol: Symbol,
        account_snapshot: AccountSnapshot,
        *,
        minimum_order_ratio: float = MINIMUM_ORDER_RATIO,
    ) -> FuturesPositionTarget:
        """State where a short position on the symbol should stand.

        Args:
            account_snapshot: The account's snapshot for this candle, read
                with `positions=True` and `balances=True`, and equity when the
                target is a share of it.
            minimum_order_ratio: Share of the total balance under which the
                difference is left alone, a tenth of a percent by default.

        Raises:
            ValueError: If the minimum is outside [0, 1].
        """
        validate_ratio("minimum_order_ratio", minimum_order_ratio)
        return FuturesPositionTarget(
            account=self,
            symbol=symbol,
            side=PositionSide.SHORT,
            account_snapshot=account_snapshot,
            minimum_order_ratio=minimum_order_ratio,
        )

    def trail_stop_loss(
        self,
        symbol: Symbol,
        distance_ratio: float,
        *,
        account_snapshot: AccountSnapshot,
    ) -> UpdatePositionTrailingStopAction:
        """Trail the stop of an open position a share of the price behind it.

        The exchange holds the trailing stop and moves it behind the best price
        the position reaches, so book the call once: a trailing stop already
        resting on the symbol is left as it is. An exchange that offers no
        trailing stop at that share raises `BotCriticalError` when the booked
        action runs, which stops the bot, and a symbol the account holds no
        position on is dropped at INFO.

        Args:
            symbol: The symbol of the open position.
            distance_ratio: Share of the best price reached the stop stays
                behind it, within (0, 1).
            account_snapshot: The account's snapshot for this candle, declared
                with `open_orders=True`.

        Raises:
            ValueError: If the ratio is outside (0, 1).
        """
        validate_trailing_ratio(distance_ratio)
        return UpdatePositionTrailingStopAction(
            exchange=self._exchange,
            symbol=symbol,
            distance_ratio=distance_ratio,
            resting=tuple(account_snapshot.trailing_stop_orders(symbol)),
        )

    def update_order(
        self,
        order: OrderProtocol,
        *,
        quantity: float | None = None,
        limit_price: float | None = None,
        trigger_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> FuturesOrderModifyAction:
        """A field the caller does not name stays as the order rests.

        Its reduce-only, trigger types and the tag of its client order id stay
        as the order rests, and so does its time in force, except on a
        triggered order, which rests GTC because the builder refuses any other
        time in force on a trigger. The replacement carries that tag in a fresh
        client order id, since a venue rejects an id it has already seen, and
        an order whose id this bot did not write, placed by hand or by another
        bot, moves with no client order id. The order's venue fields, reason
        and extra fields are not read back, and a change of shape, a limit
        order into a trigger order, is refused. The run's `[orders.entry]` and
        `[orders.exit]` settings do not apply to the replacement, its
        `[orders.stop_loss]` and `[orders.take_profit]` trigger types
        included: the brackets keep the trigger types as the order rests.

        Args:
            order: The order as the snapshot's open orders report it.
            quantity: The new size in the base currency, the part already
                filled included; None leaves the size.
            limit_price: None leaves the price as the order rests.
            trigger_price: None leaves the trigger as the order rests.
            stop_loss: None leaves the order's level; a level named on an
                order without one adds it.
            take_profit: None leaves the order's level; a level named on an
                order without one adds it.

        Raises:
            ValueError: If no field is named; the order is a stop-loss, a
                take-profit, a trailing stop, a liquidation or a market
                order; the read states no side, quantity or remaining
                quantity; a limit price or a trigger price is named on an
                order with none; the quantity does not exceed what already
                filled; a value is not finite; a price is not positive; or
                the order's client order id carries a tag the engine writes
                on its own orders, or the order cannot be built.
            BotCriticalError: If the symbol was never declared for the
                account.
        """
        _require_no_engine_tag(order)
        request = modify_request(
            order,
            quantity=quantity,
            limit_price=limit_price,
            trigger_price=trigger_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        replacement = replace(
            self._builder_of(request.order).build(),
            stop_loss=request.order.stop_loss,
            take_profit=request.order.take_profit,
        )
        return FuturesOrderModifyAction(
            exchange=self._exchange,
            symbol=replacement.symbol,
            modifications=[
                OrderReplacement(
                    order_id=order.order_id,
                    order=replacement,
                    with_client_order_id=request.order.client_order_id is not None,
                )
            ],
        )

    async def _attributed_executions(
        self,
        since: datetime,
        symbols: Iterable[Symbol],
        *,
        max_attempts: int = MAX_ATTEMPTS,
        base_delay: float = BASE_DELAY_SECONDS,
    ) -> list[Execution]:
        symbol_list = list(symbols)
        positions = await retry_on_transient(
            self._exchange.get_open_positions,
            symbol_list,
            max_attempts=max_attempts,
            base_delay=base_delay,
        )
        executions = await self._executions_since(
            since,
            symbol_list,
            max_attempts=max_attempts,
            base_delay=base_delay,
        )
        return attribute_fill_effects(executions, positions)

    def _books(self, action: BaseExchangeAction) -> bool:
        """Whether the action goes to this account's venue.

        An action a strategy defines itself may name no exchange, and then
        goes to none, so a halt never holds it back.
        """
        return getattr(action, "exchange", None) is self._exchange

    def _builder_of(self, request: OrderRequest) -> FuturesOrderBuilder:
        create = (
            self._create_buy_order
            if request.side is OrderSide.BUY
            else self._create_sell_order
        )
        builder = create(
            request.symbol,
            request.quantity,
            NO_ORDER_SETTINGS,
            reduce_only=request.reduce_only,
        )
        if request.limit_price is not None:
            builder.limit(
                request.limit_price,
                time_in_force=(
                    request.time_in_force
                    if request.trigger_price is None
                    else TimeInForce.GTC
                ),
            )
        if request.trigger_price is not None:
            builder.trigger(request.trigger_price, trigger_type=request.trigger_type)
        tag = tag_of(request.client_order_id)
        if tag is not None:
            builder.tag(tag)
        return builder

    def _close_removed_position(self, symbol: Symbol) -> ClosePositionAction:
        return ClosePositionAction(
            exchange=self._exchange, symbol=symbol, tag=FLATTEN_TAG
        )

    def _create_buy_order(
        self,
        symbol: Symbol,
        quantity: float | None,
        settings: OrderSettings,
        reduce_only: bool = False,
        reference_price: float | None = None,
    ) -> FuturesOrderBuilder:
        """Create a BUY order (opens long or closes short)."""
        return FuturesOrderBuilder(
            account=self,
            quantity=quantity,
            side=OrderSide.BUY,
            symbol=symbol,
            settings=settings,
            reduce_only=reduce_only,
            reference_price=reference_price,
        )

    def _create_sell_order(
        self,
        symbol: Symbol,
        quantity: float | None,
        settings: OrderSettings,
        reduce_only: bool = False,
        reference_price: float | None = None,
    ) -> FuturesOrderBuilder:
        """Create a SELL order (opens short or closes long)."""
        return FuturesOrderBuilder(
            account=self,
            quantity=quantity,
            side=OrderSide.SELL,
            symbol=symbol,
            settings=settings,
            reduce_only=reduce_only,
            reference_price=reference_price,
        )

    async def _executions_since(
        self,
        since: datetime,
        symbols: Iterable[Symbol],
        *,
        max_attempts: int = MAX_ATTEMPTS,
        base_delay: float = BASE_DELAY_SECONDS,
    ) -> list[Execution]:
        executions: list[Execution] = await retry_on_transient(
            self._exchange.get_executions_since,
            since,
            list(symbols),
            max_attempts=max_attempts,
            base_delay=base_delay,
        )
        return executions

    def _halts(self, symbols: Iterable[Symbol]) -> list[Halt]:
        return [
            Halt(symbol, self._venue, self.name, instrument.closure)
            for symbol in symbols
            if (instrument := self._instruments.get(symbol)) is not None
            and instrument.closure is not None
        ]

    def _instrument(self, symbol: Symbol) -> Instrument:
        try:
            return self._instruments[symbol]
        except KeyError:
            unlisted = self._unlisted.get(symbol)
            if unlisted is not None:
                raise SymbolNotListedError(unlisted) from None
            raise BotCriticalError(
                f"{self.name} trades {symbol} without declaring it: name the "
                "symbol in requirements.account.add(symbols=...) so the engine "
                "reads its trading rules and state."
            ) from None

    def _labels_resting_triggers(self) -> bool:
        return self.capabilities.labels_resting_triggers

    async def _read_instruments(self, symbols: Iterable[Symbol]) -> None:
        for symbol in symbols:
            try:
                self._instruments[symbol] = await self._exchange.get_instrument(symbol)
                self._unlisted.pop(symbol, None)
            except SymbolNotListedError as error:
                logger.debug("%s", error)
                self._instruments.pop(symbol, None)
                self._unlisted[symbol] = str(error)

    async def _snapshot(
        self,
        requirement: AccountRequirement,
        request: AccountStateRequest,
        *,
        executions_since: datetime,
    ) -> AccountSnapshot:
        state = await self._exchange.get_account_state(request, executions_since)
        return AccountSnapshot.from_state(
            state,
            account_name=self.name,
            positions_declared=requirement.positions,
            executions_declared=requirement.executions,
        )

    async def _unlisted_among(self, symbols: Iterable[Symbol]) -> frozenset[Symbol]:
        unlisted = set()
        for symbol in symbols:
            try:
                await self._exchange.get_instrument(symbol)
            except SymbolNotListedError as error:
                logger.debug("%s", error)
                unlisted.add(symbol)
        return frozenset(unlisted)


def _require_no_engine_tag(order: OrderProtocol) -> None:
    tag = tag_of(order.client_order_id)
    if tag in ENGINE_TAGS:
        raise ValueError(
            f"{order.symbol}: order {order.order_id} carries the engine's own "
            f"tag `{tag}` and is not moved by its fields"
        )
