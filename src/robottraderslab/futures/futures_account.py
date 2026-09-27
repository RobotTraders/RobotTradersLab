import asyncio
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime
from typing import Any

from robottraderslab._core import (
    BASE_DELAY_SECONDS,
    MAX_ATTEMPTS,
    AccountRequirement,
    AccountSnapshot,
    Execution,
    MarginMode,
    OrderSide,
    PositionSide,
    Symbol,
    TrackedPosition,
    attribute_fill_effects,
    finished,
    retry_on_transient,
    validate_price,
    validate_ratio,
)

from .futures_cancel_order import CancelOrderByIdAction
from .futures_cancel_orders import CancelOrdersForSymbolAction
from .futures_capabilities import FuturesCapabilities
from .futures_close_builder import FuturesCloseBuilder
from .futures_close_position import validate_closing_ratio
from .futures_exchange_protocol import FuturesExchangeProtocol
from .futures_order_batch import (
    BatchableOrder,
    FuturesOrderBatchAction,
    FuturesOrderModifyAction,
    OrderModification,
    OrderReplacement,
)
from .futures_order_builder import FuturesOrderBuilder, validate_tag
from .futures_position_target import MINIMUM_ORDER_RATIO, FuturesPositionTarget
from .futures_set_leverage import SetLeverageAction
from .futures_set_margin_mode import SetMarginModeAction
from .futures_update_position_stop_loss import UpdatePositionStopLossAction
from .futures_update_position_take_profit import UpdatePositionTakeProfitAction


class FuturesAccount:
    """A strategy's trading account, and the factory for its actions.

    Each method returns what the strategy books. An entry, an exit or a
    tracked close returns an order builder. A symbol's close returns a close
    builder. A target returns a `FuturesPositionTarget`, sized into an order
    builder. Every other method returns a finished action.
    """

    def __init__(self, exchange: FuturesExchangeProtocol, name: str | None = None):
        """Initialise the account.

        Args:
            exchange: Venue the account trades on.
            name: Identifies the account among the ones a strategy uses, and
                labels it in errors and logs. Defaults to the exchange type.
        """
        self._exchange = exchange
        self.name = name or type(exchange).__name__

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
            exchange=self._exchange, symbol=symbol, closing_ratio=closing_ratio
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
            return self._create_sell_order(symbol, quantity, reduce_only=False)
        return self._create_buy_order(symbol, quantity, reduce_only=False)

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
        return self._create_buy_order(symbol, quantity, reduce_only=False)

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
        return self._create_sell_order(symbol, quantity, reduce_only=reduce_only)

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
        """Reshape one resting order into another.

        Args:
            order_id: Exchange-assigned id of the resting order.
            order: The replacement, a built order or an order builder the
                account finishes with its `build()`.

        Raises:
            StrategyCriticalError: If the builder's `build()` refuses the order.
        """
        return self.modify_orders([OrderModification(order_id=order_id, order=order)])

    def modify_orders(
        self, modifications: Iterable[OrderModification]
    ) -> FuturesOrderModifyAction:
        """Reshape several resting orders together.

        The venue receives them as few requests as it allows, and one
        rejected modification does not stop the others. A resting order
        that filled or was cancelled in the meantime is skipped, never
        re-placed.

        Args:
            modifications: Resting orders and their replacements.

        Raises:
            ValueError: If no modifications are given.
            StrategyCriticalError: If a builder's `build()` refuses its order.
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
        self, symbol: Symbol, price: float
    ) -> UpdatePositionStopLossAction:
        """Move the stop-loss guarding an open position to a price.

        A move the venue rejects is sent once more in the same cycle, and one
        it still refuses is logged at ERROR; a move on a symbol the account
        holds no position on is dropped at INFO.

        Raises:
            ValueError: If the price is not positive or is NaN.
        """
        validate_price("price", price)
        return UpdatePositionStopLossAction(
            exchange=self._exchange,
            symbol=symbol,
            trigger_price=price,
        )

    def move_take_profit(
        self, symbol: Symbol, price: float
    ) -> UpdatePositionTakeProfitAction:
        """Move the take-profit on an open position to a price.

        A move the venue rejects is sent once more in the same cycle, and one
        it still refuses is logged at ERROR; a move on a symbol the account
        holds no position on is dropped at INFO.

        Raises:
            ValueError: If the price is not positive or is NaN.
        """
        validate_price("price", price)
        return UpdatePositionTakeProfitAction(
            exchange=self._exchange,
            symbol=symbol,
            trigger_price=price,
        )

    def place_orders(self, orders: Iterable[BatchableOrder]) -> FuturesOrderBatchAction:
        """Place several orders together.

        The venue receives them as few requests as it allows, and one
        rejected order does not stop the others.

        Args:
            orders: Built orders, or order builders the account finishes with
                their `build()`.

        Raises:
            ValueError: If no orders are given.
            StrategyCriticalError: If a builder's `build()` refuses its order.
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
        return self._create_sell_order(symbol, quantity, reduce_only=False)

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
        return self._create_buy_order(symbol, quantity, reduce_only=reduce_only)

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

    def _create_buy_order(
        self,
        symbol: Symbol,
        quantity: float | None,
        reduce_only: bool = False,
    ) -> FuturesOrderBuilder:
        """Create a BUY order (opens long or closes short)."""
        return FuturesOrderBuilder(
            account=self,
            quantity=quantity,
            side=OrderSide.BUY,
            symbol=symbol,
            reduce_only=reduce_only,
        )

    def _create_sell_order(
        self,
        symbol: Symbol,
        quantity: float | None,
        reduce_only: bool = False,
    ) -> FuturesOrderBuilder:
        """Create a SELL order (opens short or closes long)."""
        return FuturesOrderBuilder(
            account=self,
            quantity=quantity,
            side=OrderSide.SELL,
            symbol=symbol,
            reduce_only=reduce_only,
        )

    def _declared_reads(
        self,
        requirement: AccountRequirement,
        executions_since: datetime,
        report_fills: bool,
        max_attempts: int,
        base_delay: float,
    ) -> dict[str, Awaitable[Any]]:
        fetches: dict[str, Awaitable[Any]] = {}
        symbols = list(requirement.symbols)

        def declare(key: str, fn: Callable[..., Awaitable[Any]], *args: Any) -> None:
            fetches[key] = retry_on_transient(
                fn, *args, max_attempts=max_attempts, base_delay=base_delay
            )

        if requirement.open_orders:
            declare("open_orders", self._exchange.get_open_orders, symbols)
        if requirement.positions or requirement.needs_executions(report_fills):
            declare("positions", self._exchange.get_open_positions, symbols)
        if requirement.balances:
            declare("balances", self._exchange.get_balances, symbols)
            for symbol in _cross_quoted(requirement.symbols):
                declare(
                    f"rate:{symbol}", self._exchange.get_quote_conversion_rate, symbol
                )
        if requirement.margin_settings:
            declare("margin_settings", self._exchange.get_margin_settings, symbols)
        if requirement.needs_executions(report_fills):
            declare(
                "executions",
                self._exchange.get_executions_since,
                executions_since,
                symbols,
            )
        for currency in requirement.equity:
            declare(f"equity:{currency}", self._exchange.get_equity, currency)
        return fetches

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

    async def _snapshot(
        self,
        requirement: AccountRequirement,
        *,
        executions_since: datetime,
        report_fills: bool = False,
        max_attempts: int = MAX_ATTEMPTS,
        base_delay: float = BASE_DELAY_SECONDS,
    ) -> AccountSnapshot:
        """Fetch the declared state in one concurrent burst.

        Args:
            requirement: What the strategy declared for this account in `setup`.
            executions_since: Start of the executions window.
            report_fills: See `AccountRequirement.needs_executions`.
            max_attempts: Total retry attempts for transient errors, per read.
            base_delay: Base retry delay in seconds (doubled on each retry).

        Returns:
            A snapshot populated with the declared fields, plus executions
            on top when report_fills asks for them, each attributed against
            the positions the same burst read.
        """
        fetches = self._declared_reads(
            requirement, executions_since, report_fills, max_attempts, base_delay
        )
        results = dict(zip(fetches, await asyncio.gather(*fetches.values())))

        executions = results.get("executions")
        attributed = (
            attribute_fill_effects(executions, results["positions"])
            if executions is not None
            else None
        )
        equities = {
            currency: results[f"equity:{currency}"] for currency in requirement.equity
        }
        conversion_rates = {
            symbol: results[f"rate:{symbol}"]
            for symbol in _cross_quoted(requirement.symbols)
        }
        return AccountSnapshot(
            account_name=self.name,
            open_orders=results.get("open_orders"),
            positions=results.get("positions") if requirement.positions else None,
            balances=results.get("balances"),
            equities=equities or None,
            margin_settings=results.get("margin_settings"),
            conversion_rates=conversion_rates or None,
            executions=attributed,
            executions_declared=requirement.executions,
        )


def _cross_quoted(symbols: tuple[Symbol, ...]) -> tuple[Symbol, ...]:
    return tuple(symbol for symbol in symbols if symbol.quote != symbol.margin)
