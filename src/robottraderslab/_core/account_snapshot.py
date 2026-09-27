from typing import NoReturn, Protocol

from .balance import Balance
from .currency import Currency
from .exceptions import StrategyCriticalError
from .margin import MarginSettings
from .order import (
    RESTING_ENTRY_KINDS,
    VENUE_FIRED_SOURCES,
    Execution,
    FillEffect,
    OrderProtocol,
    OrderType,
    ProtectionKind,
)
from .position import PositionSnapshot
from .symbol import Symbol


class NamedAccount(Protocol):
    """An account identified by name."""

    name: str


class AccountSnapshot:
    """One account's state, frozen at the moment the candle closed, before
    the strategy books on it.

    Every accessor is a pure in-memory read. Only declared fields are
    populated, and reading an undeclared one raises `StrategyCriticalError`,
    so no read ever leaves the declared burst.
    """

    def __init__(
        self,
        *,
        account_name: str,
        open_orders: list[OrderProtocol] | None = None,
        positions: dict[Symbol, PositionSnapshot] | None = None,
        balances: dict[Currency, Balance] | None = None,
        equities: dict[Currency, float] | None = None,
        margin_settings: dict[Symbol, MarginSettings] | None = None,
        conversion_rates: dict[Symbol, float] | None = None,
        executions: list[Execution] | None = None,
        executions_declared: bool = False,
    ) -> None:
        """Initialise the snapshot.

        Args:
            account_name: Labels the account in errors.
            open_orders: None when the strategy did not declare them, which
                is what makes reading them raise. Same for the rest.
            positions: See open_orders.
            balances: See open_orders.
            equities: See open_orders.
            margin_settings: See open_orders.
            conversion_rates: See open_orders.
            executions: The executions read for the account, which the
                engine may populate on its own initiative regardless of
                whether the strategy declared them.
            executions_declared: Whether the strategy itself declared
                `executions=True`, which its own typed accessors gate on.
        """
        self._account_name = account_name
        self._open_orders = open_orders
        self._positions = positions
        self._balances = balances
        self._equities = equities
        self._margin_settings = margin_settings
        self._conversion_rates = conversion_rates
        self._executions = executions
        self._executions_declared = executions_declared

    def balance(self, currency: Currency) -> Balance:
        """Return the funds held in the currency, one of the margin currencies
        of the declared symbols.

        Raises:
            StrategyCriticalError: If balances were not declared, or no
                declared symbol carries the currency as its margin.
        """
        balances = self._declared(self._balances, "balances=True")
        if currency not in balances:
            raise StrategyCriticalError(
                f"The balance in {currency} was not read for account "
                f"'{self._account_name}': a balance is read for the margin "
                "currency of each declared symbol; include a symbol carrying it "
                "in setup() with requirements.account.add(account, symbols=[...], "
                "balances=True)."
            )
        return balances[currency]

    def entry_fills(self, symbol: Symbol) -> list[Execution]:
        """Return the limit and trigger entry fills on the symbol within the
        fetched window.
        """
        return self._fills_of_kinds(RESTING_ENTRY_KINDS, symbol)

    def equity(self, currency: Currency) -> float:
        """Return the account's equity in the currency: the funds it holds and
        the unrealised profit of its open positions together.
        """
        equities = self._declared(self._equities, f"equity=[{currency!r}]")
        if currency not in equities:
            raise StrategyCriticalError(
                f"Equity in {currency} was not declared for account "
                f"'{self._account_name}'; add it in setup() with "
                f"requirements.account.add(account, equity=[{currency!r}])."
            )
        return equities[currency]

    def liquidation_fills(self, symbol: Symbol) -> list[Execution]:
        """Return the liquidation fills on the symbol within the fetched window."""
        return self._fills_of_kind(OrderType.LIQUIDATION, symbol)

    def margin_settings(self, symbol: Symbol) -> MarginSettings:
        """Return the margin mode and leverage active on the symbol."""
        settings = self._declared(self._margin_settings, "margin_settings=True")
        if symbol not in settings:
            raise StrategyCriticalError(
                f"Margin settings for {symbol} were not declared for account "
                f"'{self._account_name}'; include the symbol in setup() with "
                "requirements.account.add(account, symbols=[...], margin_settings=True)."
            )
        return settings[symbol]

    def open_orders(self, symbol: Symbol) -> list[OrderProtocol]:
        """Return the open orders on the symbol, triggers included."""
        orders = self._declared(self._open_orders, "open_orders=True")
        return [order for order in orders if order.symbol == symbol]

    def position(self, symbol: Symbol) -> PositionSnapshot | None:
        """Return the open position on the symbol, None when the account holds none."""
        positions = self._declared(self._positions, "positions=True")
        return positions.get(symbol)

    def quote_conversion_rate(self, symbol: Symbol) -> float:
        """Return the rate converting the symbol's quote into its margin
        currency, 1 when the two are one currency.

        The rate is read with the balances, for every declared symbol quoted
        in another currency than its margin.

        Raises:
            StrategyCriticalError: If the symbol is quoted in another currency
                than its margin and balances were not declared, or the symbol
                was not.
        """
        if symbol.quote == symbol.margin:
            return 1.0
        rates = self._declared(self._conversion_rates, "balances=True")
        if symbol not in rates:
            raise StrategyCriticalError(
                f"A conversion rate for {symbol} was not read for account "
                f"'{self._account_name}'; include the symbol in setup() with "
                "requirements.account.add(account, symbols=[...], balances=True)."
            )
        return rates[symbol]

    def stop_loss_fills(self, symbol: Symbol) -> list[Execution]:
        """Return the stop-loss fills on the symbol within the fetched window."""
        return self._fills_of_kind(OrderType.STOP_LOSS, symbol)

    def stop_loss_orders(self, symbol: Symbol) -> list[OrderProtocol]:
        """Return every stop-loss resting on the symbol, empty when none rests.

        Several rest at once when each filled rung placed its own stop-loss
        beside the position's.
        """
        return self._resting_protections(OrderType.STOP_LOSS, symbol)

    def take_profit_fills(self, symbol: Symbol) -> list[Execution]:
        """Return the take-profit fills on the symbol within the fetched window."""
        return self._fills_of_kind(OrderType.TAKE_PROFIT, symbol)

    def take_profit_orders(self, symbol: Symbol) -> list[OrderProtocol]:
        """Return every take-profit resting on the symbol, empty when none rests.

        Several rest at once when each filled rung placed its own take-profit
        beside the position's.
        """
        return self._resting_protections(OrderType.TAKE_PROFIT, symbol)

    def _hold_margin_settings(self, symbol: Symbol, settings: MarginSettings) -> None:
        """The engine sets a declared margin target before the strategy books,
        so the settings it read are replaced by the ones the venue now holds.
        """
        self._declared(self._margin_settings, "margin_settings=True")[symbol] = settings

    def _declared[T](self, requested: T | None, declaration: str) -> T:
        if requested is None:
            self._undeclared(declaration)
        return requested

    def _declared_executions(self, symbol: Symbol) -> list[Execution]:
        """Reads a requirement `TrackerRequirements.add` always forces to
        declared for whichever account holds a tracker.
        """
        return [
            execution
            for execution in self._executions or []
            if execution.symbol == symbol
        ]

    def _effects_by_execution_id(self) -> dict[str, FillEffect]:
        return {
            execution.execution_id: execution.effect
            for execution in self._executions or []
            if execution.effect is not None
        }

    def _fills_of_kind(self, kind: OrderType, symbol: Symbol) -> list[Execution]:
        self._require_executions_declared()
        return self._raw_fills_of_kind(kind, symbol)

    def _fills_of_kinds(
        self, kinds: frozenset[OrderType], symbol: Symbol
    ) -> list[Execution]:
        self._require_executions_declared()
        return self._raw_fills_of_kinds(kinds, symbol)

    def _raw_fills_of_kind(self, kind: OrderType, symbol: Symbol) -> list[Execution]:
        return [
            execution
            for execution in self._executions or []
            if execution.kind == kind and execution.symbol == symbol
        ]

    def _raw_fills_of_kinds(
        self, kinds: frozenset[OrderType], symbol: Symbol
    ) -> list[Execution]:
        return [
            execution
            for execution in self._executions or []
            if execution.kind in kinds and execution.symbol == symbol
        ]

    def _require_executions_declared(self) -> None:
        if not self._executions_declared:
            self._undeclared("executions=True")

    def _resting_protections(
        self, kind: ProtectionKind, symbol: Symbol
    ) -> list[OrderProtocol]:
        return [order for order in self.open_orders(symbol) if order.kind == kind]

    def _undeclared(self, declaration: str) -> NoReturn:
        raise StrategyCriticalError(
            f"The strategy read account state it did not declare for account "
            f"'{self._account_name}'; add it in setup() with "
            f"requirements.account.add(account, {declaration})."
        )

    def _venue_fired_fills(self, symbol: Symbol) -> list[Execution]:
        """Return the fills the venue fired itself on the symbol.

        Reads whatever the engine fetched for reporting, whether or not
        the strategy itself declared executions.
        """
        return [
            execution
            for kind in VENUE_FIRED_SOURCES
            for execution in self._raw_fills_of_kind(kind, symbol)
        ]


class AccountSnapshots:
    """The declared snapshots of every account, taken for one candle."""

    def __init__(self, by_account_name: dict[str, AccountSnapshot]) -> None:
        """Initialise the collection.

        Args:
            by_account_name: One snapshot per account that declared state.
        """
        self._by_account_name = by_account_name

    def of(self, account: NamedAccount) -> AccountSnapshot:
        """Return the snapshot taken for the account.

        Raises:
            StrategyCriticalError: If the account declared no state to snapshot.
        """
        snapshot = self._by_account_name.get(account.name)
        if snapshot is None:
            raise StrategyCriticalError(
                f"No account state was declared for account '{account.name}'; "
                "declare it in setup() with requirements.account.add(account, ...)."
            )
        return snapshot

    def _effects_by_execution_id(self, account: NamedAccount) -> dict[str, FillEffect]:
        """An account may report fills without declaring state to snapshot."""
        snapshot = self._by_account_name.get(account.name)
        if snapshot is None:
            return {}
        return snapshot._effects_by_execution_id()
