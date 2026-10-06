import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import Protocol

from .account_snapshot import AccountSnapshot, AccountSnapshots, NamedAccount
from .actions.members import BaseExchangeAction
from .currency import Currency
from .exceptions import StrategyCriticalError
from .margin import MarginMode, MarginSettings
from .order import Execution, FillDescriber
from .retry import BASE_DELAY_SECONDS, MAX_ATTEMPTS
from .sizing import SizingRule, margin_currency
from .symbol import Symbol

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

    async def _snapshot(
        self,
        requirement: "AccountRequirement",
        *,
        executions_since: datetime,
        report_fills: bool = False,
        max_attempts: int = MAX_ATTEMPTS,
        base_delay: float = BASE_DELAY_SECONDS,
    ) -> "AccountSnapshot":
        """Fetch the declared state in one concurrent burst.

        Args:
            requirement: What the strategy declared for this account in `setup`.
            executions_since: Start of the executions window.
            report_fills: See `AccountRequirement.needs_executions`.
            max_attempts: Total retry attempts for transient errors, per read.
            base_delay: Base retry delay in seconds (doubled on each retry).

        Returns:
            A snapshot populated with the declared fields, plus executions
            on top when report_fills asks for them.
        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class AccountRequirement:
    """What a strategy needs from one account on every candle."""

    account: AccountProtocol
    symbols: tuple[Symbol, ...] = ()
    positions: bool = False
    open_orders: bool = False
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


class AccountRequirements:
    """Declares the account state a strategy needs."""

    def __init__(self) -> None:
        self._requirements: list[AccountRequirement] = []
        self._position_needs: dict[str, tuple[Symbol, ...]] = {}
        self._execution_needs: set[str] = set()

    def add(
        self,
        account: AccountProtocol,
        *,
        symbols: Iterable[Symbol] = (),
        positions: bool = False,
        open_orders: bool = False,
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
                these and no others.
            positions: Unlock `snapshot.position(symbol)`.
            open_orders: Unlock `snapshot.open_orders(symbol)`,
                `snapshot.stop_loss_orders(symbol)` and
                `snapshot.take_profit_orders(symbol)`.
            balances: Unlock `snapshot.balance(currency)` for the margin
                currency of each symbol, and `snapshot.quote_conversion_rate`.
            margin_settings: Unlock `snapshot.margin_settings(symbol)`.
            margin_targets: The margin mode and leverage each symbol trades
                at, which the engine sets on the venue before every candle is
                booked wherever the venue's settings differ; a target leverage
                of None leaves the leverage as the venue holds it. Declares
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
            sweep_protections: Cancel a stop-loss or take-profit left resting on
                a symbol this account holds no position on. Needs `positions`
                and `open_orders`, and does nothing without them. Pass False to
                keep protective orders the venue did not clean up.

        Raises:
            StrategyCriticalError: If an account of that name is already declared.
        """
        if any(
            declared.account.name == account.name for declared in self._requirements
        ):
            raise StrategyCriticalError(
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

    def _get_all(self) -> list[AccountRequirement]:
        return list(self._requirements)

    def _require_executions(self, account_name: str) -> None:
        """A tracker may be declared before or after its account."""
        self._execution_needs.add(account_name)
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
    declared = [
        requirement
        for requirement in requirements._get_all()
        if not requirement.is_empty(report_fills)
    ]
    snapshots = await asyncio.gather(
        *[
            requirement.account._snapshot(
                requirement,
                executions_since=executions_since,
                report_fills=report_fills,
                max_attempts=max_attempts,
                base_delay=base_delay,
            )
            for requirement in declared
        ]
    )
    return AccountSnapshots(
        {
            requirement.account.name: snapshot
            for requirement, snapshot in zip(declared, snapshots)
        }
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
