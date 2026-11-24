import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .balance import Balance
from .currency import Currency
from .margin import MarginSettings
from .order import Execution, OrderProtocol
from .position import PositionSnapshot
from .retry import BASE_DELAY_SECONDS, MAX_ATTEMPTS, retry_on_transient
from .symbol import Symbol


@dataclass(frozen=True, slots=True, kw_only=True)
class AccountStateRequest:
    """The state an account reads on every candle of a run.

    Args:
        symbols: Symbols the reads are scoped to; none means every one the
            venue lists.
        positions: Whether the open positions are read; also read for the
            executions, since each execution is measured against them.
        open_orders: Whether the resting orders are read, a trigger entry only where
            the venue lists it with the others or `triggers` asks for it.
        balances: Whether the balances are read, with the rate converting each
            symbol quoted outside its margin currency.
        margin_settings: Whether the margin mode and leverage of each symbol
            are read.
        equity: The currencies whose equity is read; empty reads none.
        executions: Whether the executions are read, once the positions have
            answered.
        max_attempts: Total attempts of each read on a transient failure; the
            engine's default attempts when not given.
        base_delay: Ceiling of the first delay between attempts, in seconds;
            the engine's default delay when not given.
    """

    symbols: tuple[Symbol, ...] = ()
    positions: bool = False
    open_orders: bool = False
    triggers: bool = False
    balances: bool = False
    margin_settings: bool = False
    equity: tuple[Currency, ...] = ()
    executions: bool = False
    max_attempts: int = MAX_ATTEMPTS
    base_delay: float = BASE_DELAY_SECONDS

    @property
    def cross_quoted(self) -> tuple[Symbol, ...]:
        return tuple(symbol for symbol in self.symbols if symbol.quote != symbol.margin)

    @property
    def reads_positions(self) -> bool:
        return self.positions or self.executions

    def retried[T](
        self, read: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> Awaitable[T]:
        """Only a transient failure is sent again; any other error raises on
        the attempt that met it.

        Args:
            read: The connector method sending the venue request.
            *args: Passed to the read unchanged on every attempt.
            **kwargs: Passed to the read unchanged on every attempt.

        Returns:
            The answer of the first attempt that does not fail transiently.

        Raises:
            ExchangeRecoverableError: The last transient failure, once the
                attempts are spent.
            ExchangeCriticalError: Any critical error the read raises, at once
                and unretried.
        """
        return retry_on_transient(
            read,
            *args,
            max_attempts=self.max_attempts,
            base_delay=self.base_delay,
            **kwargs,
        )

    async def retried_after[T](
        self, answered: Awaitable[object], read: Callable[..., Awaitable[T]], *args: Any
    ) -> T:
        """Send a venue read once another has answered, so what the first read
        saw is already in the second's answer.

        Args:
            answered: The read that answers first.
            read: The connector method sending the later venue request.
            *args: Passed to the later read unchanged on every attempt.

        Returns:
            The answer of the later read's first attempt that does not fail
            transiently.

        Raises:
            ExchangeRecoverableError: The last transient failure of either read,
                once its attempts are spent.
            ExchangeCriticalError: Any critical error either read raises.
        """
        await answered
        return await self.retried(read, *args)

    def started[T](
        self, read: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> asyncio.Future[T]:
        """Reads started one after another are all in flight before any
        answers; a read starts on the running event loop, so it is called
        from a coroutine.

        Args:
            read: The connector method sending the venue request.
            *args: Passed to the read unchanged on every attempt.
            **kwargs: Passed to the read unchanged on every attempt.

        Returns:
            The read in flight; awaiting it gives the answer of the first
            attempt that does not fail transiently.

        Raises:
            ExchangeRecoverableError: The last transient failure, once the
                attempts are spent, raised when the read is awaited.
            ExchangeCriticalError: Any critical error the read raises, raised
                when the read is awaited.
        """
        return asyncio.ensure_future(self.retried(read, *args, **kwargs))

    def started_open_orders(
        self, get_open_orders: Callable[..., Awaitable[list[OrderProtocol]]]
    ) -> asyncio.Future[list[OrderProtocol]] | None:
        """The keyword reaches the connector only when `triggers` is declared,
        so a connector whose listing takes none is called bare.

        Returns:
            The read in flight, None when `open_orders` is not requested.
        """
        if not self.open_orders:
            return None
        if self.triggers:
            return self.started(get_open_orders, self.symbols, with_triggers=True)
        return self.started(get_open_orders, self.symbols)


@dataclass(frozen=True, slots=True, kw_only=True)
class AccountState:
    """A field the request did not ask for is None, or empty for a mapping
    keyed by what the request names; the positions are present whenever the
    executions are.

    Args:
        positions: The open positions, keyed by symbol; None when the request did not read them.
        open_orders: The resting orders, a trigger entry among them where the venue lists it
            with the others or the request's `triggers` asked for it; None when the request
            did not read them.
        balances: The balance of each margin currency the venue reports; None when the request did not read them.
        margin_settings: The margin mode and leverage of each requested symbol
            the venue lists; None when the request did not read them.
        equities: The equity of each requested currency.
        executions: Oldest first, in the sequence the venue booked them, not
            yet measured against the positions; None when the request did not
            read them.
        conversion_rates: One per requested symbol quoted outside its margin
            currency, when balances were requested.
    """

    positions: dict[Symbol, PositionSnapshot] | None = None
    open_orders: list[OrderProtocol] | None = None
    balances: dict[Currency, Balance] | None = None
    margin_settings: dict[Symbol, MarginSettings] | None = None
    equities: dict[Currency, float]
    conversion_rates: dict[Symbol, float]
    executions: list[Execution] | None = None
