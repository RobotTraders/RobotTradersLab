from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from .account_snapshot import AccountSnapshots, NamedAccount
from .actions.members import ProfileTag
from .exceptions import StrategyCriticalError
from .exchange_position_tracker import ExchangePositionTracker
from .position_tracker import PositionTracker, TrackingId
from .symbol import Symbol


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackerRequirement:
    """One tracker kept in step with the venue on every candle."""

    tracker: ExchangePositionTracker
    account: NamedAccount
    ids_by_symbol: Mapping[Symbol, tuple[TrackingId, ...]]


class TrackerRequirements:
    """Declares the position tracking a strategy asks for."""

    def __init__(
        self,
        require_positions: Callable[[str, tuple[Symbol, ...]], None],
        require_executions: Callable[[str], None],
    ) -> None:
        """Use through `StrategyRequirements`.

        Args:
            require_positions: Marks an account, by name, as needing its
                positions read every candle on the given symbols.
            require_executions: Marks an account, by name, as needing its
                executions read every candle.
        """
        self._require_positions = require_positions
        self._require_executions = require_executions
        self._requirements: list[TrackerRequirement] = []
        self._claimed_symbols: set[tuple[str, Symbol]] = set()

    def add(
        self,
        account: NamedAccount,
        *,
        symbols: Mapping[TrackingId, Symbol],
        tags: Mapping[TrackingId, ProfileTag],
    ) -> PositionTracker:
        """Declare the tracking ids one account holds per symbol, and return
        their tracker.

        Declaring reads the account's positions and executions on every
        candle, so the tracker needs nothing recorded by the strategy.

        Args:
            account: Account the tracked positions are held on.
            symbols: The symbol each tracking id trades.
            tags: The tag each tracking id's orders carry, read back with
                `tag_of`; every declared id needs one, and no two ids on one
                symbol may share one.

        Raises:
            StrategyCriticalError: If a declared id has no tag, two ids on
                one symbol share one, or a symbol is already declared on the
                same account, since a fill on that symbol can belong to one
                declaration only.
        """
        ids_by_symbol = _group(symbols)
        self._claim_symbols(account.name, ids_by_symbol)
        tracker = ExchangePositionTracker.create(ids_by_symbol, tags)
        self._require_positions(account.name, tuple(ids_by_symbol))
        self._require_executions(account.name)
        self._requirements.append(
            TrackerRequirement(
                tracker=tracker, account=account, ids_by_symbol=ids_by_symbol
            )
        )
        return tracker

    def refresh_all(
        self,
        account_snapshots: AccountSnapshots,
        reported_since: datetime,
        reported_until: datetime,
    ) -> bool:
        """Runs once per candle, after the snapshot fetch and before any
        booking, so every declared record is settled by what the venue
        reports before a strategy acts on it.

        Args:
            reported_since: The open of the candle the cycle is acting on.
            reported_until: Its close. Together they decide whether a
                quantity no tracking id owns is news.

        Returns:
            Whether any tracker cannot account for a symbol's whole position
            from the candle's own read, which `widen_tracked_positions` is
            what answers. False on the candle a backtest and a settled live
            cycle run, so the hot path awaits nothing.
        """
        asking = False
        for requirement in self._requirements:
            asking |= requirement.tracker.refresh(
                account_snapshots.of(requirement.account),
                reported_since,
                reported_until,
            )
        return asking

    def _claim_symbols(
        self, account_name: str, ids_by_symbol: Mapping[Symbol, tuple[TrackingId, ...]]
    ) -> None:
        for symbol in ids_by_symbol:
            if (account_name, symbol) in self._claimed_symbols:
                raise StrategyCriticalError(
                    f"{symbol} is already tracked on account '{account_name}'; "
                    "declare every tracking id of a symbol in one "
                    "requirements.tracker.add(...)."
                )
            self._claimed_symbols.add((account_name, symbol))

    def _get_all(self) -> list[TrackerRequirement]:
        return list(self._requirements)

    def _unanchored(self) -> list[tuple[TrackerRequirement, frozenset[Symbol]]]:
        """Empty on a candle every tracked position is younger than,
        since none of them then needs a wider read.
        """
        asking = []
        for requirement in self._requirements:
            symbols = requirement.tracker.unanchored_symbols()
            if symbols:
                asking.append((requirement, symbols))
        return asking


def _group(
    symbols: Mapping[TrackingId, Symbol],
) -> dict[Symbol, tuple[TrackingId, ...]]:
    grouped: dict[Symbol, list[TrackingId]] = {}
    for tracking_id, symbol in symbols.items():
        grouped.setdefault(symbol, []).append(tracking_id)
    return {symbol: tuple(ids) for symbol, ids in grouped.items()}
