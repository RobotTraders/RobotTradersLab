import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from .account_snapshot import AccountSnapshot
from .actions.members import BRACKET_SUFFIXES, ProfileTag, tag_of
from .exceptions import BotCriticalError
from .order import Execution, OrderSide, OrderType
from .position import PositionSide, PositionSnapshot
from .position_tracker import TrackedPosition, TrackingId, position_steps
from .quantity import StepCount, to_quantity, to_step_count
from .symbol import Symbol

logger = logging.getLogger(__name__)

_PROTECTION_KINDS = frozenset({OrderType.STOP_LOSS, OrderType.TAKE_PROFIT})


@dataclass(slots=True)
class _SymbolCursor:
    latest_timestamp: datetime | None = None
    seen_at_latest: frozenset[str] = frozenset()
    net_steps: StepCount = 0
    external_steps: StepCount = 0
    external_opened_at: datetime | None = None
    external_moved_at: datetime | None = None
    unattributed_steps: StepCount = 0


class ExchangePositionTracker:
    """A tracking id's position, derived from the venue's own fills alone.

    A venue reports one netted position per symbol however many tracking ids
    share it, so each id's share is resolved from the tag its own orders
    carry. A stop firing between candles, a hand on the exchange, or a
    resumed process with no local state all leave it correct.
    """

    def __init__(
        self,
        ids_by_symbol: Mapping[Symbol, tuple[TrackingId, ...]],
        id_by_symbol_tag: Mapping[tuple[Symbol, str], TrackingId],
        netted: bool,
    ) -> None:
        """Build through `create`, which validates and inverts the tag map first.

        Args:
            id_by_symbol_tag: The tracking id a tag resolves to on a symbol.
            netted: Whether a fill taking a symbol's net through flat clears
                every id of that symbol.
        """
        self._ids_by_symbol = ids_by_symbol
        self._id_by_symbol_tag = id_by_symbol_tag
        self._netted = netted
        self._exposure: dict[TrackingId, StepCount] = {}
        self._cursors = {symbol: _SymbolCursor() for symbol in ids_by_symbol}
        self._unanchored: set[Symbol] = set()
        self._reporting: set[Symbol] = set()

    @classmethod
    def create(
        cls,
        ids_by_symbol: Mapping[Symbol, tuple[TrackingId, ...]],
        tags: Mapping[TrackingId, ProfileTag],
        *,
        netted: bool = False,
    ) -> "ExchangePositionTracker":
        """Rejects a bad tag before any tracker built from it exists.

        Args:
            tags: The tag each declared id's own orders carry.
            netted: Whether the declared ids on a symbol are the legs of the
                symbol's one netted position, which ends when the venue's net
                reaches flat.

        Raises:
            BotCriticalError: If tags leave a declared id untagged, or
                one is shared by two tracking ids on one symbol.
        """
        _require_every_id_tagged(ids_by_symbol, tags)
        return cls(ids_by_symbol, _id_by_symbol_tag(ids_by_symbol, tags), netted)

    def get(self, tracking_id: TrackingId) -> TrackedPosition | None:
        """Return the exposure the venue's own fills have folded onto the id."""
        steps = self._exposure.get(tracking_id, 0)
        if steps == 0:
            return None
        side = PositionSide.LONG if steps > 0 else PositionSide.SHORT
        return TrackedPosition(side, to_quantity(abs(steps)))

    def refresh(
        self,
        account_snapshot: AccountSnapshot,
        reported_since: datetime,
        reported_until: datetime,
    ) -> bool:
        """Return whether a symbol is left the fold cannot account for.

        Runs once per candle, after the snapshot fetch and before any
        booking, so every tracked id reflects what the venue reports before
        a strategy acts on it. Work is proportional to what is new: an
        execution already folded is never folded again. A symbol is settled
        on every candle, since a delisting settlement or a fill landing after
        the read moves the venue's position with no new execution; its new
        executions are folded only when it has some.

        Args:
            account_snapshot: The account's state of the candle, which must
                declare the account's positions (`positions=True` on
                `requirements.account.add`).
            reported_since: The open of the earliest candle the cycle closed.
            reported_until: The close of the candles the cycle closed. A
                quantity no tracking id owns is reported at WARNING when a
                fill of those candles moved it, and at INFO while it stands,
                so the engine carries nothing between cycles; a fill landing
                after the close is the next cycle's to report.

        Returns:
            Whether a symbol is left the fold cannot account for, the
            question `unanchored_symbols` then answers in full.

        Raises:
            BotCriticalError: If `account_snapshot` declared no positions.
        """
        new_by_symbol = self._new_by_symbol(account_snapshot._account_executions())
        positions = account_snapshot._account_positions()
        cursors = self._cursors
        for symbol in self._ids_by_symbol:
            cursor = cursors[symbol]
            new_executions = new_by_symbol.get(symbol)
            if new_executions:
                self._walk(symbol, cursor, new_executions)
            self._settle(symbol, cursor, positions.get(symbol))
        for symbol in self._reporting:
            if symbol not in self._unanchored:
                _report(symbol, self._cursors[symbol], reported_since, reported_until)
        return bool(self._unanchored)

    def settle_unlisted(self, symbol: Symbol) -> None:
        """Settle a symbol the venue's listing lacks and the account holds no position on.

        A delisting settles every position on the contract and no fill of it
        arrives afterwards, so the profiles on the symbol hold nothing
        whatever fills the window carries. The venue's listing keeps the
        symbol settled, so it stays unanchored.

        Args:
            symbol: A symbol the tracker was declared on, which the account's
                position read shows no position on.

        Raises:
            KeyError: If the tracker was not declared on the symbol.
        """
        self._clear(symbol, self._cursors[symbol])
        logger.warning(
            f"{symbol}: the venue lists no contract for it, so its position is "
            "settled and no profile holds any of it"
        )

    def unanchored_symbols(self) -> frozenset[Symbol]:
        """Return the symbols whose fills do not reach back to a flat moment.

        The fold accounts for every unit of the venue's position only once it
        starts from a moment the symbol held nothing, so a symbol naming
        itself here is asking for a read reaching further back.
        """
        return frozenset(self._unanchored)

    def widen(
        self,
        account_snapshot: AccountSnapshot,
        executions: Sequence[Execution],
        symbols: Iterable[Symbol],
        read_since: datetime,
        reported_since: datetime,
        reported_until: datetime,
    ) -> None:
        """Fold the longer read of each unanchored symbol and report it.

        A symbol whose read is widened says nothing until it has been,
        since the wider fold is what decides how much of it has an owner.

        Args:
            account_snapshot: The account's state of the candle, which must
                declare the account's positions (`positions=True` on
                `requirements.account.add`).
            executions: The executions of the longer read, whatever symbol,
                from which each widened symbol takes its own.
            symbols: The unanchored symbols the venue still lists, each one the
                tracker was declared on.
            read_since: Where that read starts, which is what a position
                still unattributed after it was opened before.
            reported_since: See `refresh`.
            reported_until: See `refresh`.

        Raises:
            BotCriticalError: If `account_snapshot` declared no positions.
            KeyError: If the tracker was not declared on a symbol.
        """
        for symbol in symbols:
            cursor = self._refold(symbol, account_snapshot, executions)
            _report(symbol, cursor, reported_since, reported_until)
            if cursor.unattributed_steps != 0:
                _log_unattributed(symbol, cursor.unattributed_steps, read_since)

    def _attribute(
        self, symbol: Symbol, execution: Execution, before: StepCount, after: StepCount
    ) -> TrackingId | None:
        tracking_id = self._owner(symbol, execution)
        if (
            tracking_id is None
            and is_unlabelled_protection(execution)
            and not _passes_through_flat(before, after)
        ):
            return self._closed_id(symbol, after - before)
        return tracking_id

    def _clear(self, symbol: Symbol, cursor: _SymbolCursor) -> None:
        for tracking_id in self._ids_by_symbol[symbol]:
            self._exposure.pop(tracking_id, None)
        cursor.external_steps = 0
        cursor.external_opened_at = None

    def _closed_id(self, symbol: Symbol, signed_steps: StepCount) -> TrackingId | None:
        """Return the id a protection fill closed, None when none alone did.

        A protection reduces the position it closes and never turns it, so
        an id holding less than the fill closes did not close it alone.
        """
        closable: TrackingId | None = None
        closable_count = 0
        matching: TrackingId | None = None
        matching_count = 0
        for tracking_id in self._ids_by_symbol[symbol]:
            held = self._exposure.get(tracking_id, 0)
            if held * signed_steps < 0:
                closable = tracking_id
                closable_count += 1
                if held == -signed_steps:
                    matching = tracking_id
                    matching_count += 1
        if closable_count == 1 and closable is not None:
            if abs(self._exposure[closable]) >= abs(signed_steps):
                return closable
            return None
        if matching_count == 1:
            return matching
        return None

    def _fold(
        self,
        symbol: Symbol,
        cursor: _SymbolCursor,
        execution: Execution,
        signed_steps: StepCount,
    ) -> None:
        """Fold one fill into the exposure of the id that owns it.

        A tag names the declared id whose order produced the fill, so the
        fill moves that id's exposure alone, whatever the venue calls its
        effect: ids offsetting each other net the venue flat while each keeps
        its own side. A fill carrying no declared tag was placed outside these
        declarations, and what it opens is held outside them too.

        A close nets the whole symbol, so an untagged fill taking the venue
        through flat leaves nothing standing behind it, and what it leaves in
        front of it was opened outside these declarations like the fill
        itself. An untagged fill that neither finds nor leaves a quantity held
        outside, a liquidation closing only what the declared ids held, has
        nothing outside to report.

        A stop-loss or take-profit the venue fires carries no label of its
        own, so one taking the venue's position short of flat belongs to the
        id holding what it closes: the only one on the side it closes, or the
        one whose exposure it equals when several hold it.

        On a netted declaration the ids are the legs of one position, so a tagged
        fill taking the venue to flat or through it ends that position too: every
        id is cleared, and what a crossing fill leaves in front of flat is the
        opening of the next position, held by the fill's own id.

        Folding in whole steps, the unit a fill is never finer than, keeps
        two fills that cancel exactly from leaving any residue behind.
        """
        before = cursor.net_steps
        after = before + signed_steps
        cursor.net_steps = after
        tracking_id = self._attribute(symbol, execution, before, after)
        held_outside = cursor.external_steps != 0
        if _passes_through_flat(before, after) and (
            tracking_id is None or self._netted
        ):
            self._restart_at_flat(symbol, cursor, tracking_id, after)
        elif tracking_id is not None:
            self._exposure[tracking_id] = (
                self._exposure.get(tracking_id, 0) + signed_steps
            )
            return
        else:
            cursor.external_steps += signed_steps
        self._report_outside_move(symbol, cursor, execution, held_outside)

    def _new_by_symbol(
        self, executions: Iterable[Execution]
    ) -> dict[Symbol, list[Execution]]:
        """Return each symbol's executions the fold has not yet seen.

        One pass over a read in any order, each execution weighed against
        its own symbol's cursor, since neither the venue's order nor a
        connector's concatenation of several reads says where the new ones
        are.

        Fills share exact timestamps, so the cursor also tracks the ids
        already folded at its latest timestamp.
        """
        found: dict[Symbol, list[Execution]] = {}
        cursors = self._cursors
        for execution in executions:
            cursor = cursors.get(execution.symbol)
            if cursor is None:
                continue
            boundary = cursor.latest_timestamp
            if boundary is not None:
                timestamp = execution.timestamp
                if timestamp < boundary or (
                    timestamp == boundary
                    and execution.execution_id in cursor.seen_at_latest
                ):
                    continue
            found.setdefault(execution.symbol, []).append(execution)
        return found

    def _owner(self, symbol: Symbol, execution: Execution) -> TrackingId | None:
        tag = tag_of(execution.client_order_id)
        if tag is None:
            return None
        return self._id_by_symbol_tag.get((symbol, tag))

    def _refold(
        self,
        symbol: Symbol,
        account_snapshot: AccountSnapshot,
        executions: Sequence[Execution],
    ) -> _SymbolCursor:
        cursor = _SymbolCursor()
        self._cursors[symbol] = cursor
        self._unanchored.discard(symbol)
        self._reporting.discard(symbol)
        self._clear(symbol, cursor)
        self._walk(symbol, cursor, on_symbol(executions, symbol))
        self._settle(symbol, cursor, account_snapshot.position(symbol))
        return cursor

    def _report_outside_move(
        self,
        symbol: Symbol,
        cursor: _SymbolCursor,
        execution: Execution,
        held_outside: bool,
    ) -> None:
        if not held_outside and cursor.external_steps == 0:
            return
        cursor.external_moved_at = execution.timestamp
        self._reporting.add(symbol)
        if cursor.external_steps == 0:
            cursor.external_opened_at = None
        elif cursor.external_opened_at is None:
            cursor.external_opened_at = execution.timestamp

    def _restart_at_flat(
        self,
        symbol: Symbol,
        cursor: _SymbolCursor,
        owner: TrackingId | None,
        remainder: StepCount,
    ) -> None:
        self._clear(symbol, cursor)
        if owner is None:
            cursor.external_steps = remainder
        elif remainder != 0:
            self._exposure[owner] = remainder

    def _settle(
        self, symbol: Symbol, cursor: _SymbolCursor, position: PositionSnapshot | None
    ) -> None:
        unattributed = position_steps(position) - cursor.net_steps
        if unattributed == cursor.unattributed_steps:
            return
        cursor.unattributed_steps = unattributed
        if unattributed == 0:
            self._unanchored.discard(symbol)
        else:
            self._unanchored.add(symbol)

    def _walk(
        self, symbol: Symbol, cursor: _SymbolCursor, new_executions: Sequence[Execution]
    ) -> None:
        if not new_executions:
            return
        folded = sorted(
            new_executions,
            key=lambda execution: (execution.timestamp, execution.execution_id),
        )
        for execution in folded:
            self._fold(symbol, cursor, execution, signed_steps(execution))
        _advance_cursor(cursor, folded)


def is_unlabelled_protection(execution: Execution) -> bool:
    """Return whether the execution is a protection fill with no tag of its own.

    A protection of a bracket order names no tracking id either, so it counts
    as untagged.
    """
    if execution.kind not in _PROTECTION_KINDS:
        return False
    tag = tag_of(execution.client_order_id)
    return tag is None or tag in BRACKET_SUFFIXES


def on_symbol(executions: Iterable[Execution], symbol: Symbol) -> list[Execution]:
    """Return the executions that fill the symbol."""
    return [execution for execution in executions if execution.symbol == symbol]


def signed_steps(execution: Execution) -> StepCount:
    """Return the execution's quantity in whole steps, negative on a sell."""
    steps = to_step_count(execution.quantity)
    return steps if execution.side == OrderSide.BUY else -steps


def _require_every_id_tagged(
    ids_by_symbol: Mapping[Symbol, tuple[TrackingId, ...]],
    tags: Mapping[TrackingId, ProfileTag],
) -> None:
    declared_ids = {
        tracking_id for ids in ids_by_symbol.values() for tracking_id in ids
    }
    missing = sorted(declared_ids - tags.keys())
    if missing:
        raise BotCriticalError(
            f"Tracking id(s) {missing} have no tag; give every id declared in "
            "requirements.tracker.add(tags=...) its own tag."
        )


def _id_by_symbol_tag(
    ids_by_symbol: Mapping[Symbol, tuple[TrackingId, ...]],
    tags: Mapping[TrackingId, ProfileTag],
) -> dict[tuple[Symbol, str], TrackingId]:
    """Invert the tag map to the id a tag resolves to on a symbol.

    A fill carries its symbol, so a tag only has to be unique among the
    ids declared on that symbol.
    """
    inverted: dict[tuple[Symbol, str], TrackingId] = {}
    for symbol, tracking_ids in ids_by_symbol.items():
        for tracking_id in tracking_ids:
            tag = tags[tracking_id]
            collision = inverted.get((symbol, tag))
            if collision is not None:
                raise BotCriticalError(
                    f"Tracking ids '{collision}' and '{tracking_id}' share the "
                    f"tag '{tag}' on {symbol}; give every tracking id on a "
                    "symbol its own tag."
                )
            inverted[(symbol, tag)] = tracking_id
    return inverted


def _report(
    symbol: Symbol,
    cursor: _SymbolCursor,
    reported_since: datetime,
    reported_until: datetime,
) -> None:
    """Log what a symbol's quantity held outside the declarations did.

    The engine carries nothing between cycles, so the candle a fill lands
    in is what makes the report of what it moved fire once: a fill booked
    after the close, before the cycle read it, belongs to the next candle.
    """
    moved_at = cursor.external_moved_at
    if moved_at is not None and reported_since <= moved_at < reported_until:
        _log_external(symbol, cursor)
    elif cursor.external_steps != 0:
        _log_standing_external(symbol, cursor)


def _log_external(symbol: Symbol, cursor: _SymbolCursor) -> None:
    """Log that the quantity held outside the declarations moved.

    An external share and the moment it was opened stand or go together,
    so the symbol is clear exactly when there is no moment left to name.
    """
    opened_at = cursor.external_opened_at
    if opened_at is None:
        logger.warning(
            f"{symbol}: the quantity no order of this strategy opened is gone"
        )
        return
    logger.warning(
        f"{symbol}: {to_quantity(cursor.external_steps):+g} of the venue's position "
        f"was opened by an order this strategy did not place, on {_stamp(opened_at)}"
    )


def _log_standing_external(symbol: Symbol, cursor: _SymbolCursor) -> None:
    logger.info(
        f"{symbol}: {to_quantity(cursor.external_steps):+g} of the venue's position "
        "is held outside this strategy"
    )


def _log_unattributed(symbol: Symbol, steps: StepCount, read_since: datetime) -> None:
    logger.warning(
        f"{symbol}: {to_quantity(steps):+g} of the venue's position was opened "
        f"before {_stamp(read_since)}, as far back as the fill history is read, "
        "and is left unattributed"
    )


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M UTC")


def _passes_through_flat(before: StepCount, after: StepCount) -> bool:
    if before == 0:
        return False
    return after == 0 or (before > 0) != (after > 0)


def _advance_cursor(cursor: _SymbolCursor, folded: Sequence[Execution]) -> None:
    newest_timestamp = folded[-1].timestamp
    folded_at_newest = {
        execution.execution_id
        for execution in folded
        if execution.timestamp == newest_timestamp
    }
    if newest_timestamp == cursor.latest_timestamp:
        cursor.seen_at_latest |= folded_at_newest
    else:
        cursor.seen_at_latest = frozenset(folded_at_newest)
    cursor.latest_timestamp = newest_timestamp
