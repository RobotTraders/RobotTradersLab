from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, Protocol

from robottraderslab._core import (
    Balance,
    Currency,
    Execution,
    MarginMode,
    MarginSettings,
    OrderModifyRequest,
    OrderProtocol,
    OrderRequest,
    OrderSide,
    PlacedOrder,
    PositionSnapshot,
    StopLoss,
    Symbol,
    TakeProfit,
    TimeInForce,
)

from .futures_capabilities import FuturesCapabilities


class FuturesExchangeProtocol(Protocol):
    """What the engine needs from a futures venue."""

    capabilities: FuturesCapabilities = FuturesCapabilities()
    placement_reserve_rate: float = 0.0

    def placement_requirement_rate(self, leverage: float) -> float:
        """Share of an order's value the venue locks at placement at a
        leverage, the margin included; a venue holding back nothing beyond
        the margin locks the margin alone.
        """
        return 1 / leverage

    async def get_balances(self, symbols: Iterable[Symbol]) -> dict[Currency, Balance]:
        """Return the balance of each margin currency the venue reports.

        Args:
            symbols: Symbols the read is scoped to; none means every one.
        """
        ...

    async def get_equity(self, currency: Currency) -> float: ...

    async def get_margin_settings(
        self, symbols: Iterable[Symbol]
    ) -> dict[Symbol, MarginSettings]:
        """Return the margin mode and leverage currently active for each symbol.

        Returns:
            One entry per requested symbol. Leverage is None when the venue
            defines no per-symbol leverage in the active margin mode.
        """
        ...

    async def get_open_orders(self, symbols: Iterable[Symbol]) -> list[OrderProtocol]:
        """Return the resting orders of the account, triggers included.

        Args:
            symbols: Symbols the read is scoped to; none means every one.
        """
        ...

    async def get_open_positions(
        self, symbols: Iterable[Symbol]
    ) -> dict[Symbol, PositionSnapshot]:
        """Return the open positions of the account, keyed by symbol.

        Args:
            symbols: Symbols the read is scoped to; none means every one.
        """
        ...

    async def get_quote_conversion_rate(self, symbol: Symbol) -> float:
        """Return the rate converting the symbol's quote currency into the margin currency.

        Returns:
            Multiplier applied to quote-denominated amounts. 1.0 when the pair
            is quoted in the margin currency (linear contracts).
        """
        ...

    async def get_executions_since(
        self, since: datetime, symbols: Iterable[Symbol]
    ) -> list[Execution]:
        """Return every execution the venue booked against the account.

        Args:
            since: Return only executions booked at or after this timestamp.
            symbols: Symbols the read is scoped to; none means every one.

        Returns:
            Executions oldest first, in the sequence the venue booked them.
            Reading a position's history depends on that sequence, and several
            executions can share one instant, so the venue's own order between
            them is authoritative. Lookback may be capped by the venue's
            history retention.
        """
        ...

    async def set_leverage(self, symbol: Symbol, leverage: float) -> None:
        """Set the leverage for a symbol.

        Args:
            leverage: Must be >= 1.0.

        Raises:
            ValueError: If leverage < 1.0
        """

    async def set_margin_mode(self, symbol: Symbol, margin_mode: MarginMode) -> None:
        """Set the margin mode for a symbol."""

    async def place_market_order(
        self,
        symbol: Symbol,
        side: OrderSide,
        quantity: float,
        reduce_only: bool = False,
        stop_loss: StopLoss | None = None,
        take_profit: TakeProfit | None = None,
        trigger_price: float | None = None,
        client_order_id: str | None = None,
        **kwargs: Any,
    ) -> PlacedOrder:
        """Place a market order.

        Args:
            reduce_only: Whether the order only reduces an existing position.
            client_order_id: Caller-assigned order id, unchanged across
                retried attempts so the venue rejects a duplicate submission.
            **kwargs: Additional keyword arguments (e.g., reason for analytics)
        """

    async def place_limit_order(
        self,
        symbol: Symbol,
        side: OrderSide,
        quantity: float,
        price: float,
        reduce_only: bool = False,
        stop_loss: StopLoss | None = None,
        take_profit: TakeProfit | None = None,
        trigger_price: float | None = None,
        client_order_id: str | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        **kwargs: Any,
    ) -> PlacedOrder:
        """Place a limit order.

        Args:
            reduce_only: Whether the order only reduces an existing position.
            client_order_id: Caller-assigned order id, unchanged across
                retried attempts so the venue rejects a duplicate submission.
            time_in_force: How long the order may wait on the book; a
                `POST_ONLY` order that would fill at once is refused instead.
            **kwargs: Additional keyword arguments (e.g., reason for analytics)

        Raises:
            ExchangeRecoverableError: If the venue refuses the order,
                including a `POST_ONLY` one that would take liquidity.
        """

    async def place_orders(
        self, requests: Sequence[OrderRequest]
    ) -> list[PlacedOrder | None]:
        """Place several orders together, as few requests as the venue allows.

        A venue rejecting one order does not stop the others: the rejected
        order is logged and reported as None. A transient failure raises so
        the caller retries the whole batch, which the unchanged client order
        ids keep from placing duplicates. A venue without a batch endpoint
        inherits one placement per request from FuturesExchangeBase.

        Returns:
            The placed order for each request, in request order, with None
            where the venue rejected that request.
        """
        ...

    async def modify_orders(
        self, requests: Sequence[OrderModifyRequest]
    ) -> list[PlacedOrder | None]:
        """Reshape resting orders into replacements, as few requests as the venue allows.

        A target that has left the book is skipped and reported as None,
        never re-placed: its disappearance means it filled or was cancelled,
        and the caller sees the outcome on its next snapshot. Failure
        semantics match place_orders. A venue without a safe modification
        endpoint inherits one cancel and re-place per request from
        FuturesExchangeBase.

        Args:
            requests: Resting orders and the replacements they should become.

        Returns:
            The resulting order for each request, in request order, with
            None where the target was gone or the venue rejected the
            replacement.
        """
        ...

    async def cancel_order_by_id(self, symbol: Symbol, order_id: str) -> None:
        """Cancel a single open order by its order id.

        Args:
            order_id: Exchange-assigned order id
        """

    async def cancel_orders_for_symbol(self, symbol: Symbol) -> None:
        """Cancel all open orders for the symbol."""

    async def update_position_stop_loss(
        self, symbol: Symbol, trigger_price: float
    ) -> PlacedOrder:
        """Update the stop-loss trigger price of an open position.

        Returns:
            The protection order the venue now holds, under the id it reports
            among its open orders.

        Raises:
            NoOpenPositionError: If the account holds no position on the symbol.
        """

    async def update_position_take_profit(
        self, symbol: Symbol, trigger_price: float
    ) -> PlacedOrder:
        """Update the take-profit trigger price of an open position.

        Returns:
            The protection order the venue now holds, under the id it reports
            among its open orders.

        Raises:
            NoOpenPositionError: If the account holds no position on the symbol.
        """
