from datetime import UTC, datetime, timedelta

from .account_requirements import AccountProtocol
from .account_snapshot import AccountSnapshots
from .retry import BASE_DELAY_SECONDS, MAX_ATTEMPTS
from .strategy_requirements import StrategyRequirements

ATTRIBUTION_REACH = timedelta(days=90)


async def widen_tracked_positions(
    requirements: StrategyRequirements,
    account_snapshots: AccountSnapshots,
    *,
    executions_since: datetime,
    reported_since: datetime,
    reported_until: datetime,
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY_SECONDS,
) -> None:
    """A tracker accounts for a symbol's whole position only from a moment
    the symbol held nothing, and the candle's own read reaches back no
    further than the strategy's lookback, so a position older than that
    leaves the tracker asking for more. A symbol the venue's listing lacks
    has had its position settled by the delisting, so no read widens for it.

    Args:
        executions_since: Where the candle's own read starts. The second
            read never starts later, so a strategy whose lookback already
            reaches past the attribution span keeps every fill it had.
        reported_since: The open of the earliest candle the cycle closed.
        reported_until: The close of the candles the cycle closed. Together
            they decide whether a quantity no tracking id owns is news.
        max_attempts: Total retry attempts for transient errors, per read.
        base_delay: Base retry delay in seconds (doubled on each retry).
    """
    asking = requirements.tracker._unanchored()
    accounts = _accounts_by_name(requirements)
    since = min(executions_since, datetime.now(UTC) - ATTRIBUTION_REACH)
    for requirement, symbols in asking:
        account = accounts[requirement.account.name]
        unlisted = await account._unlisted_among(symbols)
        for symbol in unlisted:
            requirement.tracker.settle_unlisted(symbol)
        listed = symbols - unlisted
        if not listed:
            continue
        executions = await account._executions_since(
            since,
            listed,
            max_attempts=max_attempts,
            base_delay=base_delay,
        )
        requirement.tracker.widen(
            account_snapshots.of(requirement.account),
            executions,
            listed,
            since,
            reported_since,
            reported_until,
        )


def _accounts_by_name(
    requirements: StrategyRequirements,
) -> dict[str, AccountProtocol]:
    return {
        declared.account.name: declared.account
        for declared in requirements.account._get_all()
    }
