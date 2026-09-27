from typing import Any, Self

from robottraderslab._core import (
    ActionBuilder,
    OnFillRead,
    Symbol,
    TimeInForce,
    validate_price,
)

from .futures_close_position import ClosePositionAction
from .futures_exchange_protocol import FuturesExchangeProtocol
from .futures_order_builder import validate_tag


class FuturesCloseBuilder(ActionBuilder[ClosePositionAction]):
    """The close of a position on a symbol, sized from what the venue holds
    when the action runs, so it carries no sizing, no trigger and no
    protection.
    """

    def __init__(
        self, *, exchange: FuturesExchangeProtocol, symbol: Symbol, closing_ratio: float
    ) -> None:
        """Initialise the builder.

        Args:
            closing_ratio: Share of the position the order closes.
        """
        self._exchange = exchange
        self._symbol = symbol
        self._closing_ratio = closing_ratio
        self._limit_price: float | None = None
        self._time_in_force = TimeInForce.GTC
        self._reason: str | None = None
        self._tag: str | None = None
        self._extra_fields: dict[str, Any] | None = None
        self._on_filled: OnFillRead | None = None

    def build(self) -> ClosePositionAction:
        """Return the finished action. The bookkeeper finishes a builder
        handed to it with this call, at the moment it is handed over.
        """
        return ClosePositionAction(
            exchange=self._exchange,
            symbol=self._symbol,
            closing_ratio=self._closing_ratio,
            limit_price=self._limit_price,
            time_in_force=self._time_in_force,
            reason=self._reason,
            tag=self._tag,
            extra_fields=self._extra_fields,
            on_filled=self._on_filled,
        )

    def extra_fields(self, **fields: Any) -> Self:
        """Record custom fields on the backtest's fill; a venue receives none.

        Args:
            **fields: Merges into any fields an earlier call gave.
        """
        if self._extra_fields is None:
            self._extra_fields = {}
        self._extra_fields.update(fields)
        return self

    def limit(
        self, price: float, *, time_in_force: TimeInForce = TimeInForce.GTC
    ) -> Self:
        """Rest the close at a limit price, reduce-only; without it the close
        goes at market.

        Args:
            time_in_force: How long the order may wait on the book, one of
                `TimeInForce`.

        Raises:
            ValueError: If price is not positive or is NaN.
        """
        validate_price("price", price)
        self._limit_price = price
        self._time_in_force = time_in_force
        return self

    def reason(self, reason: str) -> Self:
        """Explain why the position was closed, for later analysis.

        Args:
            reason: What triggered the close (e.g. "MA crossover", "time exit").
        """
        self._reason = reason
        return self

    def tag(self, tag: str) -> Self:
        """Label the close so a later run can recognise it as its own, in the
        client order id where `tag_of` reads it.

        Raises:
            ValueError: If the tag is empty.
        """
        validate_tag(tag)
        self._tag = tag
        return self

    def when_filled(self, callback: OnFillRead) -> Self:
        """Register a callback invoked once the close's fill is read.

        Args:
            callback: Receives the placed order and its fill, None if it did
                not fill; never called when the symbol was flat.
        """
        self._on_filled = callback
        return self
