import asyncio
from collections.abc import Collection, Iterator, Mapping
from dataclasses import replace
from types import MappingProxyType

from robottraderslab_core import Symbol
from robottraderslab_core.engine import (
    BASE_DELAY_SECONDS,
    MAX_ATTEMPTS,
    AccountProtocol,
    AccountRequirement,
    execute_setting,
)
from robottraderslab_core.exchanges import MarginSettings
from robottraderslab_core.strategies import (
    AccountRequirements,
    AccountSnapshot,
    AccountSnapshots,
)


async def hold_margin_targets(
    requirements: AccountRequirements,
    snapshots: AccountSnapshots,
    *,
    closed: Mapping[AccountProtocol, Collection[Symbol]] = MappingProxyType({}),
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY_SECONDS,
) -> None:
    """Run before the strategy books on the snapshots.

    Each snapshot then carries what the venue holds: the target where its
    setter ran, the settings read where the venue refused it. A symbol the
    connector left out of the read carries no target.

    Args:
        closed: The symbols of each account no setter runs on, whatever
            their settings; the snapshot keeps the settings it read.
        max_attempts: Total retry attempts for transient errors, per setter.
        base_delay: Base retry delay in seconds (doubled on each retry).

    Raises:
        ExchangeCriticalError: If the venue is unusable.
    """
    for requirement in requirements._get_all():
        if not requirement.margin_targets:
            continue
        snapshot = snapshots.of(requirement.account)
        drifted = [
            (symbol, target)
            for symbol, target in _held_targets(requirement, snapshot, closed)
            if _differs(snapshot.margin_settings(symbol), target)
        ]
        if drifted:
            await asyncio.gather(
                *[
                    _hold_margin_target(
                        requirement.account,
                        snapshot,
                        symbol,
                        target,
                        max_attempts,
                        base_delay,
                    )
                    for symbol, target in drifted
                ]
            )


def assume_margin_targets(
    requirements: AccountRequirements,
    snapshots: AccountSnapshots,
    *,
    closed: Mapping[AccountProtocol, Collection[Symbol]] = MappingProxyType({}),
) -> None:
    """Write each declared target on its snapshot in memory, with no setter run.

    A dry run reads the settings the venue holds and sends nothing, so the
    strategy books on the snapshot with each declared target applied: the
    target where one is declared, and the settings read where the target leaves
    the leverage as the venue holds it. A symbol the connector left out of the
    read carries no target.

    Args:
        closed: The symbols of each account whose snapshot keeps the settings
            it read, whatever their targets.
    """
    for requirement in requirements._get_all():
        if not requirement.margin_targets:
            continue
        snapshot = snapshots.of(requirement.account)
        for symbol, target in _held_targets(requirement, snapshot, closed):
            held = snapshot.margin_settings(symbol)
            snapshot._hold_margin_settings(
                symbol,
                replace(
                    held,
                    margin_mode=target.margin_mode,
                    leverage=(
                        held.leverage if target.leverage is None else target.leverage
                    ),
                ),
            )


def _held_targets(
    requirement: AccountRequirement,
    snapshot: AccountSnapshot,
    closed: Mapping[AccountProtocol, Collection[Symbol]],
) -> Iterator[tuple[Symbol, MarginSettings]]:
    """The declared targets of the symbols the snapshot read the settings of.

    The symbols in `closed` are left out.
    """
    return (
        (symbol, target)
        for symbol, target in requirement.margin_targets.items()
        if symbol not in closed.get(requirement.account, ())
        and snapshot._has_margin_settings(symbol)
    )


def _differs(held: MarginSettings, target: MarginSettings) -> bool:
    return held.margin_mode != target.margin_mode or (
        target.leverage is not None and held.leverage != target.leverage
    )


async def _hold_margin_target(
    account: AccountProtocol,
    snapshot: AccountSnapshot,
    symbol: Symbol,
    target: MarginSettings,
    max_attempts: int,
    base_delay: float,
) -> None:
    """The margin mode is set first, and the leverage after any change of it,
    since a venue may keep a leverage per margin mode and the one read belongs
    to the mode left.
    """
    held = snapshot.margin_settings(symbol)
    mode_changed = held.margin_mode != target.margin_mode and await execute_setting(
        account.set_margin_mode(symbol, target.margin_mode),
        max_attempts=max_attempts,
        base_delay=base_delay,
    )
    if mode_changed:
        held = replace(held, margin_mode=target.margin_mode)
    if (
        target.leverage is not None
        and (mode_changed or held.leverage != target.leverage)
        and await execute_setting(
            account.set_leverage(symbol, target.leverage),
            max_attempts=max_attempts,
            base_delay=base_delay,
        )
    ):
        held = replace(held, leverage=target.leverage)
    snapshot._hold_margin_settings(symbol, held)
