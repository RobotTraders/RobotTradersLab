import asyncio
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import Protocol

from .account_snapshot import AccountSnapshot, AccountSnapshots, NamedAccount
from .account_state import AccountStateRequest
from .actions.members import BaseExchangeAction
from .currency import Currency
from .exceptions import BotCriticalError
from .margin import MarginMode, MarginSettings
from .order import Execution, FillDescriber
from .position import PositionSnapshot
from .retry import BASE_DELAY_SECONDS, MAX_ATTEMPTS
from .sizing import SizingRule, margin_currency
from .symbol import Symbol
from .trading_status import Halt

_NO_MARGIN_TARGETS: Mapping[Symbol, MarginSettings] = MappingProxyType({})


class AccountProtocol(NamedAccount, Protocol):
    """The part of an account the engine uses: its name, snapshot, cancels and
    margin setters.
    """

    def cancel_order(self, symbol: Symbol, order_id: str) -> BaseExchangeAction:
        """Return an action cancelling one open order by its exchange id.

        Args:
            symbol: Symbol the order sits on.
            order_id: Exchange-assigned id of the order to cancel.

        Returns:
            The action, to book.
        """
        ...

    def set_leverage(self, symbol: Symbol, leverage: float) -> BaseExchangeAction:
        """Return an action setting the account's leverage on the symbol.

        Args:
            symbol: Symbol the leverage applies to.
            leverage: Leverage to trade the symbol at, at least 1.

        Returns:
            The action, to execute.
        """
        ...

    def set_margin_mode(
        self, symbol: Symbol, margin_mode: MarginMode
    ) -> BaseExchangeAction:
        """Return an action setting how margin backs the account's positions on
        the symbol.

        Args:
            symbol: Symbol the margin mode applies to.
            margin_mode: Margin mode to trade the symbol in.

        Returns:
            The action, to execute.
        """
        ...

    async def _attributed_executions(
        self,
        since: datetime,
        symbols: Iterable[Symbol],
        *,
        max_attempts: int = MAX_ATTEMPTS,
        base_delay: float = BASE_DELAY_SECONDS,
    ) -> list[Execution]:
        """Fetch the executions the venue booked since a timestamp, each
        carrying what it did to the position.

        Each effect is measured against the position the stream ended on.

        Args:
            since: Start of the window to read.
            symbols: Symbols the read is scoped to.
            max_attempts: Total retry attempts for transient errors, per read.
            base_delay: Base retry delay in seconds.

        Returns:
            The executions, in the sequence the venue booked them.
        """
        ...

    def _books(self, action: BaseExchangeAction) -> bool:
        """Whether the action goes to the venue this account trades on.

        Args:
            action: An action a strategy booked.
        """
        ...

    def _close_removed_position(self, symbol: Symbol) -> BaseExchangeAction:
        """Return an action closing the position the account holds on a symbol no
        profile declares any more, reduce-only at market, under an id that
        names the bot.

        Returns:
            Nothing is placed by this call; the caller books the action.
        """
        ...

    async def _executions_since(
        self,
        since: datetime,
        symbols: Iterable[Symbol],
        *,
        max_attempts: int = MAX_ATTEMPTS,
        base_delay: float = BASE_DELAY_SECONDS,
    ) -> list[Execution]:
        """Fetch the executions the venue booked since a timestamp.

        Args:
            since: Start of the window to read. A moment older than the
                venue's own history reads back as far as it still reports.
            symbols: Symbols the read is scoped to; none means every one.
            max_attempts: Total retry attempts for transient errors.
            base_delay: Base retry delay in seconds (doubled on each retry).

        Returns:
            The executions, in the sequence the venue booked them.
        """
        ...

    def _halts(self, symbols: Iterable[Symbol]) -> list[Halt]:
        """Return the halts among the symbols given.

        A halted symbol is one whose venue lets the bot send less than it
        would on an open one, as `_read_instruments` read it.

        Args:
            symbols: Symbols the account declared.
        """
        ...

    def _labels_resting_triggers(self) -> bool:
        """Whether a trigger entry resting on the venue keeps the client order
        id it was placed with.

        Returns:
            False when the venue lists every resting trigger of a symbol as
            the bot's own.
        """
        ...

    def _quantity_denial(
        self, symbol: Symbol, quantity: float, price: float | None
    ) -> str | None:
        """Return why the venue's rules refuse a plain order of this quantity.

        Args:
            symbol: Symbol the order trades.
            quantity: The order's size, in the base currency.
            price: Price the order's value is measured at; None measures only
                the quantity step.

        Returns:
            None when the order is accepted; the refusal when the quantity is
            under one step, whatever the price, the venue lists no contract
            for the symbol or, given a price, the order is worth less than
            the minimum order value.

        Raises:
            BotCriticalError: If the symbol was not declared to the account.
        """
        ...

    async def _read_instruments(self, symbols: Iterable[Symbol]) -> None:
        """Every order the account builds is held to the rules read here. A
        symbol the venue lists no contract for is held as unlisted.
        """
        ...

    async def _read_positions(
        self,
        symbols: Iterable[Symbol],
        *,
        max_attempts: int = MAX_ATTEMPTS,
        base_delay: float = BASE_DELAY_SECONDS,
    ) -> dict[Symbol, PositionSnapshot]:
        """Fetch the positions the venue holds on the symbols, in one read.

        Args:
            symbols: Symbols the read is scoped to.
            max_attempts: Total retry attempts for transient errors.
            base_delay: Base retry delay in seconds (doubled on each retry).

        Returns:
            The open positions by symbol; a symbol with none is absent.

        Raises:
            ExchangeRecoverableError: If the read still fails once its attempts
                are spent.
            ExchangeCriticalError: If the venue is unusable.
        """
        ...

    async def _snapshot(
        self,
        requirement: "AccountRequirement",
        request: AccountStateRequest,
        *,
        executions_since: datetime,
    ) -> "AccountSnapshot":
        """Fetch the declared state as one unit, answered in full or raised.

        Args:
            requirement: What the strategy declared for this account in `setup`.
            request: The reads the declaration takes, fixed for the run.
            executions_since: Executions booked before this instant are left
                out, and the venue may cap how far back the window reaches.

        Returns:
            A snapshot populated with the declared fields, plus executions
            on top when the request asks for them.

        Raises:
            ExchangeRecoverableError: If a read still fails once its attempts
                are spent.
            ExchangeCriticalError: If the venue is unusable on any read.
        """
        ...

    async def _unlisted_among(self, symbols: Iterable[Symbol]) -> frozenset[Symbol]:
        """The symbols among those given that the venue's listing lacks.

        A delisting settles every position on its contract and no fill of it
        arrives afterwards, yet the listing says nothing of what the account
        still holds on the symbol: its position read does.

        Args:
            symbols: Symbols to test against the venue's listing.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class AccountRequirement:
    """What a strategy needs from one account on every candle."""

    account: AccountProtocol
    symbols: tuple[Symbol, ...] = ()
    positions: bool = False
    open_orders: bool = False
    triggers: bool = False
    balances: bool = False
    margin_settings: bool = False
    margin_targets: Mapping[Symbol, MarginSettings] = _NO_MARGIN_TARGETS
    executions: bool = False
    notify_entry_fills: bool = False
    describe_fills: FillDescriber | None = None
    equity: tuple[Currency, ...] = ()
    sweep_protections: bool = True

    def is_empty(self, report_fills: bool) -> bool:
        """Return True when the account declared no state to snapshot.

        Entry-fill notifications are reconciled after booking, never read
        into the snapshot, so they do not count here.

        Args:
            report_fills: See `needs_executions`.
        """
        return not (
            self.positions
            or self.open_orders
            or self.balances
            or self.margin_settings
            or self.equity
            or self.needs_executions(report_fills)
        )

    def needs_executions(self, report_fills: bool) -> bool:
        """Return True when the account's executions should be read from the venue.

        Args:
            report_fills: Whether a fill subscriber is configured.
        """
        return self.executions or bool(report_fills and self.symbols)

    def state_request(
        self, report_fills: bool, max_attempts: int, base_delay: float
    ) -> AccountStateRequest:
        """The reads of one candle, which every later candle of the run repeats.

        Args:
            report_fills: Whether a fill subscriber is configured, which reads
                the executions of an account declaring symbols.
            max_attempts: Total attempts of each read on a transient failure.
            base_delay: Ceiling of the first delay between attempts, in seconds.
        """
        executions = self.needs_executions(report_fills)
        return AccountStateRequest(
            symbols=self.symbols,
            positions=self.positions,
            open_orders=self.open_orders,
            triggers=self.triggers,
            balances=self.balances,
            margin_settings=self.margin_settings,
            equity=self.equity,
            executions=executions,
            max_attempts=max_attempts,
            base_delay=base_delay,
        )


class AccountRequirements:
    """Declares the account state a strategy needs."""

    def __init__(self) -> None:
        self._requirements: list[AccountRequirement] = []
        self._position_needs: dict[str, tuple[Symbol, ...]] = {}
        self._execution_needs: set[str] = set()
        self._state_reads: dict[
            tuple[bool, int, float],
            list[tuple[AccountRequirement, AccountStateRequest]],
        ] = {}

    def add(
        self,
        account: AccountProtocol,
        *,
        symbols: Iterable[Symbol] = (),
        positions: bool = False,
        open_orders: bool = False,
        triggers: bool = False,
        balances: bool = False,
        margin_settings: bool = False,
        margin_targets: Mapping[Symbol, MarginSettings] = MappingProxyType({}),
        executions: bool = False,
        notify_entry_fills: bool = False,
        describe_fills: FillDescriber | None = None,
        equity: Iterable[Currency] = (),
        sizing: Iterable[SizingRule] = (),
        sweep_protections: bool = True,
    ) -> None:
        """Declare an account and the state to read from it every candle.

        Args:
            account: Account the state is read from.
            symbols: Symbols this account trades. Per-symbol state covers
                these and no others, and the account holds their trading
                rules; an order on any other symbol is a strategy error that
                stops the run.
            positions: Unlock `snapshot.position(symbol)`.
            open_orders: Unlock `snapshot.open_orders(symbol)`,
                `snapshot.stop_loss_orders(symbol)` and
                `snapshot.take_profit_orders(symbol)`.
            triggers: List the resting trigger entries among
                `snapshot.open_orders(symbol)`, as `OrderType.TRIGGER` orders.
            balances: Unlock `snapshot.balance(currency)` for the margin
                currency of each symbol, and `snapshot.quote_conversion_rate`.
            margin_settings: Unlock `snapshot.margin_settings(symbol)`.
            margin_targets: The margin mode and leverage each symbol trades
                at, which the engine sets on the venue before every candle is
                booked wherever the venue's settings differ, except on a symbol
                its venue closed to every order, on which nothing is set; a
                target leverage of None leaves the leverage as the venue holds
                it. Declares
                `margin_settings` and adds the symbols to `symbols`.
            executions: Unlock `snapshot.entry_fills(symbol)`,
                `snapshot.stop_loss_fills(symbol)`, `snapshot.take_profit_fills(symbol)`
                and `snapshot.liquidation_fills(symbol)`, covering the loaded
                candle window.
            notify_entry_fills: Report the limit and trigger entries the
                exchange fires between runs, which no engine action executed.
            describe_fills: Gives the reason reported with each fill the venue
                produced on its own, as `FillDescriber` states.
            equity: Currencies to unlock in `snapshot.equity`.
            sizing: The rules the strategy sizes with, which declare what they
                read: `balances`, equity for the margin currencies of
                `symbols`, or `margin_settings`.
            sweep_protections: Cancel a stop-loss, take-profit or trailing stop
                left resting on a symbol this account holds no position on. Needs `positions`
                and `open_orders`, and does nothing without them. Pass False to
                keep protective orders the venue did not clean up.

        Raises:
            BotCriticalError: If an account of that name is already declared,
                or `triggers` is declared without `open_orders`.
        """
        _require_open_orders_for_triggers(account.name, triggers, open_orders)
        if any(
            declared.account.name == account.name for declared in self._requirements
        ):
            raise BotCriticalError(
                f"An account named '{account.name}' is already declared. Declare "
                "each account once, and give accounts distinct names."
            )
        symbols = tuple(symbols)
        sizing = tuple(sizing)
        if margin_targets:
            symbols = tuple(dict.fromkeys([*symbols, *margin_targets]))
        requirement = AccountRequirement(
            account=account,
            symbols=symbols,
            positions=positions,
            open_orders=open_orders,
            triggers=triggers,
            balances=balances or any(rule.reads_balances for rule in sizing),
            margin_settings=margin_settings
            or bool(margin_targets)
            or any(rule.reads_margin_settings for rule in sizing),
            margin_targets=margin_targets,
            executions=executions,
            notify_entry_fills=notify_entry_fills,
            describe_fills=describe_fills,
            equity=_with_sizing_equity(equity, sizing, symbols),
            sweep_protections=sweep_protections,
        )
        tracked = self._position_needs.get(account.name)
        if tracked is not None:
            requirement = _reading_positions(requirement, tracked)
        if account.name in self._execution_needs:
            requirement = _reading_executions(requirement)
        self._requirements.append(requirement)
        self._state_reads.clear()

    def _declared_reads(
        self, report_fills: bool, max_attempts: int, base_delay: float
    ) -> list[tuple[AccountRequirement, AccountStateRequest]]:
        """A run reads with the same settings on every candle."""
        key = (report_fills, max_attempts, base_delay)
        reads = self._state_reads.get(key)
        if reads is None:
            reads = [
                (
                    requirement,
                    requirement.state_request(report_fills, max_attempts, base_delay),
                )
                for requirement in self._requirements
                if not requirement.is_empty(report_fills)
            ]
            self._state_reads[key] = reads
        return reads

    def _get_all(self) -> list[AccountRequirement]:
        return list(self._requirements)

    def _halts(self) -> list[Halt]:
        """Return every symbol a declared account's venue halted.

        The halts are read with the instruments.
        """
        return [
            halt
            for requirement in self._requirements
            for halt in requirement.account._halts(requirement.symbols)
        ]

    def _reading_holdings_of(
        self, symbols_by_account: Mapping[AccountProtocol, Collection[Symbol]]
    ) -> "AccountRequirements":
        """A copy to read the holdings of the symbols with, and nothing else.

        Each account among the keys is asked for its positions and its open
        orders, trigger entries included, on its own symbols alone. The
        declaration it copies is left as the strategy made it.

        Args:
            symbols_by_account: The symbols whose holdings are read, by account.
        """
        return _declaring(
            [
                _reading_holdings(requirement, symbols_by_account[requirement.account])
                for requirement in self._requirements
                if requirement.account in symbols_by_account
            ]
        )

    def _widened_for_exit_only(
        self,
        exit_only_symbols: frozenset[Symbol],
        halted: Collection[AccountProtocol],
    ) -> "AccountRequirements":
        """A copy to read the state with, in which an account declaring a
        symbol in exit-only mode, or an account among `halted`, reads its
        positions and its open orders, trigger entries included, whatever it
        declared. The declaration it copies is left as the strategy made it.

        Args:
            exit_only_symbols: Symbols run in exit-only mode on every account
                that declares them.
            halted: Accounts whose venue reports a symbol they declare closed
                to new positions.
        """
        if not exit_only_symbols and not halted:
            return self
        return _declaring(
            [
                (
                    _reading_for_exit_only(requirement)
                    if requirement.account in halted
                    or not exit_only_symbols.isdisjoint(requirement.symbols)
                    else requirement
                )
                for requirement in self._requirements
            ]
        )

    def _require_executions(self, account_name: str) -> None:
        """A tracker may be declared before or after its account."""
        self._execution_needs.add(account_name)
        self._state_reads.clear()
        for index, requirement in enumerate(self._requirements):
            if requirement.account.name == account_name:
                self._requirements[index] = _reading_executions(requirement)

    def _require_positions(
        self, account_name: str, symbols: tuple[Symbol, ...]
    ) -> None:
        """A tracker may be declared before or after its account, so the need
        is kept and merged into whichever declaration exists or arrives.
        """
        combined = self._position_needs.get(account_name, ()) + symbols
        self._position_needs[account_name] = combined
        self._state_reads.clear()
        for index, requirement in enumerate(self._requirements):
            if requirement.account.name == account_name:
                self._requirements[index] = _reading_positions(requirement, combined)


async def fetch_account_snapshots(
    requirements: AccountRequirements,
    *,
    executions_since: datetime,
    report_fills: bool = False,
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY_SECONDS,
) -> AccountSnapshots:
    """Fetch every declared account's state, all accounts at once.

    Args:
        requirements: Account declarations collected in `setup`.
        executions_since: Start of the executions window.
        report_fills: See `AccountRequirement.needs_executions`.
        max_attempts: Total retry attempts for transient errors, per read.
        base_delay: Base retry delay in seconds (doubled on each retry).

    Returns:
        The snapshots, keyed by account.
    """
    declared = requirements._declared_reads(report_fills, max_attempts, base_delay)
    snapshots = await asyncio.gather(
        *[
            requirement.account._snapshot(
                requirement, request, executions_since=executions_since
            )
            for requirement, request in declared
        ]
    )
    return AccountSnapshots(
        {
            requirement.account.name: snapshot
            for (requirement, _), snapshot in zip(declared, snapshots)
        }
    )


def _require_open_orders_for_triggers(
    account_name: str, triggers: bool, open_orders: bool
) -> None:
    if triggers and not open_orders:
        raise BotCriticalError(
            f"Account '{account_name}' declares `triggers=True` without "
            "`open_orders=True`; the trigger entries are listed among the "
            "open orders, so declare both."
        )


def _with_sizing_equity(
    equity: Iterable[Currency],
    sizing: Iterable[SizingRule],
    symbols: tuple[Symbol, ...],
) -> tuple[Currency, ...]:
    declared = tuple(equity)
    if not any(rule.reads_equity for rule in sizing):
        return declared
    return tuple(
        dict.fromkeys([*declared, *(margin_currency(symbol) for symbol in symbols)])
    )


def _reading_positions(
    requirement: AccountRequirement, tracked: tuple[Symbol, ...]
) -> AccountRequirement:
    """A tracked symbol the account did not declare would read as a flat
    venue and wipe its record, so the tracked symbols join the account's own.
    """
    merged = tuple(dict.fromkeys(requirement.symbols + tracked))
    return replace(requirement, positions=True, symbols=merged)


def _reading_executions(requirement: AccountRequirement) -> AccountRequirement:
    return replace(requirement, executions=True)


def _declaring(requirement_list: list[AccountRequirement]) -> AccountRequirements:
    declared = AccountRequirements()
    declared._requirements = requirement_list
    return declared


def _reading_holdings(
    requirement: AccountRequirement, symbols: Collection[Symbol]
) -> AccountRequirement:
    return replace(
        _reading_for_exit_only(requirement),
        symbols=tuple(sorted(symbols)),
        balances=False,
        margin_settings=False,
        margin_targets=_NO_MARGIN_TARGETS,
        executions=False,
        equity=(),
    )


def _reading_for_exit_only(requirement: AccountRequirement) -> AccountRequirement:
    return replace(requirement, positions=True, open_orders=True, triggers=True)
