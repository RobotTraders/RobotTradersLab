import asyncio
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, Protocol

from robottraderslab._core import (
    NO_VENUE_PARAMS,
    AccountState,
    AccountStateRequest,
    Balance,
    Currency,
    Execution,
    Instrument,
    MarginMode,
    MarginSettings,
    OrderModifyRequest,
    OrderProtocol,
    OrderRequest,
    OrderSide,
    PlacedOrder,
    PositionSnapshot,
    Quote,
    StopLoss,
    Symbol,
    TakeProfit,
    TimeInForce,
    TriggerType,
    VenueParams,
)

from .futures_capabilities import FuturesCapabilities


class FuturesExchangeProtocol(Protocol):
    """What the engine and a program running beside it need from a futures venue."""

    capabilities: FuturesCapabilities = FuturesCapabilities()
    placement_reserve_rate: float = 0.0

    def placement_requirement_rate(self, leverage: float) -> float:
        """Share of an order's value the venue locks at placement at a
        leverage, the margin included; a venue holding back nothing beyond
        the margin locks the margin alone.
        """
        return 1 / leverage

    async def get_account_state(
        self, request: AccountStateRequest, executions_since: datetime
    ) -> AccountState:
        """The state a request names is read as one unit.

        Every read of the unit starts together, and the executions are read
        only once the positions have answered, so a position the state shows
        closed by a stop-loss carries that stop-loss among its executions. A
        connector serving several reads from one venue request overrides this
        and keeps both guarantees.

        Args:
            request: The attempts and delay each venue request retries a
                transient failure with ride in it.
            executions_since: Executions booked before this instant are left
                out, and the venue may cap how far back the window reaches.

        Returns:
            The state, a field the request did not ask for left None, or empty
            for a mapping; the positions are read whenever the executions are.

        Raises:
            ExchangeRecoverableError: If a read still fails once its attempts
                are spent.
            ExchangeCriticalError: If the venue is unusable on any read, as
                when it refuses the account's credentials.
        """
        return await read_account_state(self, request, executions_since)

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
            One entry per requested symbol the venue lists. Leverage is None when the venue
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
        self, symbols: Iterable[Symbol], *, with_protections: bool = False
    ) -> dict[Symbol, PositionSnapshot]:
        """Return the open positions of the account, keyed by symbol.

        Args:
            symbols: Symbols the read is scoped to; none means every one.
            with_protections: Whether each position carries the levels of
                its stop-loss and take-profit. A venue that lists the
                protections apart from the position reads them in a request
                beside it; False leaves both levels None and sends no such
                request.
        """
        ...

    async def get_instrument(self, symbol: Symbol) -> Instrument:
        """Return the limits the venue enforces on the symbol.

        A program sizing its own orders reads them here.

        Raises:
            SymbolNotListedError: If the venue lists no contract for the symbol.
        """
        ...

    async def get_price_increment(self, symbol: Symbol, price: float) -> float:
        """Return the smallest price step the venue accepts around the price given.

        A venue whose tick follows the price answers for the price given.

        Args:
            symbol: A symbol the venue lists.
            price: The price the step is read at, above zero; a venue whose
                tick is fixed does not use it.

        Returns:
            The step, in the symbol's quote currency, always above zero.

        Raises:
            SymbolNotListedError: If the venue lists no contract for the symbol.
        """
        ...

    async def get_quote(self, symbol: Symbol) -> Quote:
        """Return the best bid, the best ask and the last traded price of the symbol
        at the moment of the read.

        A program deciding on a price reads it here; the engine decides on
        candle closes and never calls it.

        Returns:
            A quote whose last price is None on a symbol the venue lists but
            records no trade of.

        Raises:
            SymbolNotListedError: If the venue lists no contract for the symbol.
            ExchangeRecoverableError: If the book holds no bid or no ask.
        """
        ...

    async def round_price(self, symbol: Symbol, price: float) -> float:
        """Return the price rounded the way the venue accepts it on the symbol.

        Raises:
            SymbolNotListedError: If the venue lists no contract for the symbol.
        """
        ...

    async def round_quantity(self, symbol: Symbol, quantity: float) -> float:
        """Return the quantity cut down to what the venue accepts on the symbol,
        never above the one given.

        Raises:
            SymbolNotListedError: If the venue lists no contract for the symbol.
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
        params: VenueParams = NO_VENUE_PARAMS,
        trigger_type: TriggerType = TriggerType.DEFAULT,
        **kwargs: Any,
    ) -> PlacedOrder:
        """Place a market order.

        Args:
            reduce_only: Whether the order only reduces an existing position.
            client_order_id: Caller-assigned order id, unchanged across
                retried attempts so the venue rejects a duplicate submission.
            params: Venue fields by connector name; the connector merges its
                own through `with_venue_params`.
            trigger_type: The price the trigger level watches; `DEFAULT`
                leaves the choice to the connector.
            **kwargs: Additional keyword arguments (e.g., reason for analytics)

        Raises:
            BotCriticalError: If a venue field names one the connector
                sets from the order itself, or the venue offers no trigger on
                a trigger type the order carries; no request is sent.
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
        params: VenueParams = NO_VENUE_PARAMS,
        trigger_type: TriggerType = TriggerType.DEFAULT,
        **kwargs: Any,
    ) -> PlacedOrder:
        """Place a limit order.

        Args:
            reduce_only: Whether the order only reduces an existing position.
            client_order_id: Caller-assigned order id, unchanged across
                retried attempts so the venue rejects a duplicate submission.
            time_in_force: How long the order may wait on the book; a
                `POST_ONLY` order that would fill at once is refused instead.
            params: Venue fields by connector name; the connector merges its
                own through `with_venue_params`.
            trigger_type: The price the trigger level watches; `DEFAULT`
                leaves the choice to the connector.
            **kwargs: Additional keyword arguments (e.g., reason for analytics)

        Raises:
            BotCriticalError: If a venue field names one the connector
                sets from the order itself, or the venue offers no trigger on
                a trigger type the order carries; no request is sent.
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
        self,
        symbol: Symbol,
        trigger_price: float,
        trigger_type: TriggerType = TriggerType.DEFAULT,
        client_order_id: str | None = None,
    ) -> PlacedOrder:
        """Update the stop-loss trigger price of an open position.

        Args:
            trigger_type: The price the stop-loss watches; `DEFAULT` leaves the
                choice to the connector.
            client_order_id: Id the order the venue places carries where the
                venue keeps one on it; it stays the same when the update is
                sent again.

        Returns:
            The protection order the venue now holds, under the id it reports
            among its open orders.

        Raises:
            NoOpenPositionError: If the account holds no position on the symbol.
            BotCriticalError: If the venue offers no trigger on the
                trigger type; no request is sent.
        """

    async def place_trailing_stop_order(
        self,
        symbol: Symbol,
        distance: float,
        activation_price: float | None = None,
        quantity: float | None = None,
    ) -> PlacedOrder:
        """Place a trailing stop on an open position: an order the venue trails
        behind the price.

        Args:
            distance: How far, in the quote currency, the order stays behind
                the best price reached.
            activation_price: Price the market reaches before the trail
                starts; without it the trail starts at once.
            quantity: Size the order closes, in the units of the position's
                quantity; none means the whole position.

        Returns:
            The trailing stop the venue now holds, under the id it
            reports among its open orders, at its activation price when it
            has one.

        Raises:
            NoOpenPositionError: If the account holds no position on the symbol.
            BotCriticalError: If the venue offers no trailing stop,
                takes only an activation price and none was given, takes none
                and one was given, or cannot express the distance; no request
                is sent.
        """

    async def update_position_take_profit(
        self,
        symbol: Symbol,
        trigger_price: float,
        trigger_type: TriggerType = TriggerType.DEFAULT,
        client_order_id: str | None = None,
    ) -> PlacedOrder:
        """Update the take-profit trigger price of an open position.

        Args:
            trigger_type: The price the take-profit watches; `DEFAULT` leaves the
                choice to the connector.
            client_order_id: Id the order the venue places carries where the
                venue keeps one on it; it stays the same when the update is
                sent again.

        Returns:
            The protection order the venue now holds, under the id it reports
            among its open orders.

        Raises:
            NoOpenPositionError: If the account holds no position on the symbol.
            BotCriticalError: If the venue offers no trigger on the
                trigger type; no request is sent.
        """


async def read_account_state(
    exchange: FuturesExchangeProtocol,
    request: AccountStateRequest,
    executions_since: datetime,
) -> AccountState:
    """The read a connector gets from `get_account_state` unless it overrides
    that method.

    Args:
        exchange: The connector whose venue methods are called, one per read.
        request: The reads to send; each retries with the request's attempts
            and delay.
        executions_since: Executions booked before this instant are left out,
            and the venue may cap how far back the window reaches.

    Returns:
        The requested fields answered, the others None, or empty for a
        mapping.

    Raises:
        ExchangeRecoverableError: If a read still fails once its attempts are
            spent.
        ExchangeCriticalError: If the venue is unusable on any read, as when it
            refuses the account's credentials.
    """
    symbols = request.symbols
    open_orders_read = (
        request.started(exchange.get_open_orders, symbols)
        if request.open_orders
        else None
    )
    positions_read, executions_read = _positions_then_executions(
        exchange, request, executions_since
    )
    balances_read, rates_reads = _wallet_reads(exchange, request)
    margin_settings_read = (
        request.started(exchange.get_margin_settings, symbols)
        if request.margin_settings
        else None
    )
    equities_reads = {
        currency: request.started(exchange.get_equity, currency)
        for currency in request.equity
    }
    await asyncio.gather(
        *_in_flight(
            open_orders_read,
            positions_read,
            balances_read,
            margin_settings_read,
            executions_read,
            *rates_reads.values(),
            *equities_reads.values(),
        )
    )
    return AccountState(
        positions=_answer(positions_read),
        open_orders=_answer(open_orders_read),
        balances=_answer(balances_read),
        margin_settings=_answer(margin_settings_read),
        equities={currency: read.result() for currency, read in equities_reads.items()},
        conversion_rates={
            symbol: read.result() for symbol, read in rates_reads.items()
        },
        executions=_answer(executions_read),
    )


def _positions_then_executions(
    exchange: FuturesExchangeProtocol,
    request: AccountStateRequest,
    executions_since: datetime,
) -> tuple[
    asyncio.Future[dict[Symbol, PositionSnapshot]] | None,
    asyncio.Future[list[Execution]] | None,
]:
    """A position seen closed by a stop-loss carries that stop-loss among its
    executions.
    """
    if not request.reads_positions:
        return None, None
    positions_read = request.started(exchange.get_open_positions, request.symbols)
    if not request.executions:
        return positions_read, None
    executions_read = asyncio.ensure_future(
        request.retried_after(
            positions_read,
            exchange.get_executions_since,
            executions_since,
            request.symbols,
        )
    )
    return positions_read, executions_read


def _wallet_reads(
    exchange: FuturesExchangeProtocol, request: AccountStateRequest
) -> tuple[
    asyncio.Future[dict[Currency, Balance]] | None,
    dict[Symbol, asyncio.Future[float]],
]:
    """A balance in one currency is worth reading with the rate converting
    each symbol quoted in another.
    """
    if not request.balances:
        return None, {}
    balances_read = request.started(exchange.get_balances, request.symbols)
    rates_reads = {
        symbol: request.started(exchange.get_quote_conversion_rate, symbol)
        for symbol in request.cross_quoted
    }
    return balances_read, rates_reads


def _in_flight(*reads: asyncio.Future[Any] | None) -> list[asyncio.Future[Any]]:
    return [read for read in reads if read is not None]


def _answer[T](read: asyncio.Future[T] | None) -> T | None:
    return None if read is None else read.result()
