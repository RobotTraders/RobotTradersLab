import asyncio
from dataclasses import replace

from .account_requirements import AccountProtocol, AccountRequirements
from .account_snapshot import AccountSnapshot, AccountSnapshots
from .actions import execute_setting
from .margin import MarginSettings
from .retry import BASE_DELAY_SECONDS, MAX_ATTEMPTS
from .symbol import Symbol


async def hold_margin_targets(
    requirements: AccountRequirements,
    snapshots: AccountSnapshots,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY_SECONDS,
) -> None:
    """Run before the strategy books on the snapshots.

    Each snapshot then carries what the venue holds: the target where its
    setter ran, the settings read where the venue refused it.

    Args:
        max_attempts: Total retry attempts for transient errors, per setter.
        base_delay: Base retry delay in seconds (doubled on each retry).

    Raises:
        ExchangeCriticalError: If the venue is unusable.
        StrategyCriticalError: If a margin target names a symbol the venue
            cannot trade.
    """
    for requirement in requirements._get_all():
        if not requirement.margin_targets:
            continue
        snapshot = snapshots.of(requirement.account)
        drifted = [
            (symbol, target)
            for symbol, target in requirement.margin_targets.items()
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
