import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from .account_snapshot import AccountSnapshot
from .actions.members import ProfileTag, tag_of
from .exceptions import StrategyCriticalError
from .order import Execution, OrderSide
from .position import PositionSide
from .position_tracker import TrackedPosition, TrackingId, signed_position
from .quantity import StepCount, to_quantity, to_step_count
from .symbol import Symbol

logger = logging.getLogger(__name__)


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
    ) -> None:
        """Build through `create`, which validates and inverts the tag map first.

        Args:
            id_by_symbol_tag: The tracking id a tag resolves to on a symbol.
        """
        self._ids_by_symbol = ids_by_symbol
        self._id_by_symbol_tag = id_by_symbol_tag
        self._exposure: dict[TrackingId, StepCount] = {}
        self._cursors: dict[Symbol, _SymbolCursor] = {}
        self._unanchored: set[Symbol] = set()
        self._reporting: set[Symbol] = set()

    @classmethod
    def create(
        cls,
        ids_by_symbol: Mapping[Symbol, tuple[TrackingId, ...]],
        tags: Mapping[TrackingId, ProfileTag],
    ) -> "ExchangePositionTracker":
        """Rejects a bad tag before any tracker built from it exists.

        Args:
            tags: The tag each declared id's own orders carry.

        Raises:
            StrategyCriticalError: If tags leave a declared id untagged, or
                one is shared by two tracking ids on one symbol.
        """
        _require_every_id_tagged(ids_by_symbol, tags)
        return cls(ids_by_symbol, _id_by_symbol_tag(ids_by_symbol, tags))

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
        """Runs once per candle, after the snapshot fetch and before any
        booking, so every tracked id reflects what the venue reports before
        a strategy acts on it. Work is proportional to what is new: an
        execution already folded is never walked again.

        Args:
            reported_since: The open of the candle the cycle is acting on.
            reported_until: Its close. A quantity no tracking id owns is
                reported at WARNING when a fill of that candle moved it, and
                at INFO while it stands, so the engine carries nothing
                between cycles; a fill landing after the close is that
                candle's successor's to report.

        Returns:
            Whether a symbol is left the fold cannot account for, the
            question `unanchored_symbols` then answers in full.
        """
        for symbol in self._ids_by_symbol:
            self._refresh_symbol(symbol, account_snapshot)
        for symbol in self._reporting:
            if symbol not in self._unanchored:
                _report(symbol, self._cursors[symbol], reported_since, reported_until)
        return bool(self._unanchored)

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
        read_since: datetime,
        reported_since: datetime,
        reported_until: datetime,
    ) -> None:
        """A symbol whose read is widened says nothing until it has been,
        since the wider fold is what decides how much of it has an owner.

        Args:
            executions: The executions of the longer read, whatever symbol,
                from which each unanchored symbol takes its own.
            read_since: Where that read starts, which is what a position
                still unattributed after it was opened before.
            reported_since: See `refresh`.
            reported_until: See `refresh`.
        """
        for symbol in self.unanchored_symbols():
            cursor = self._refold(symbol, account_snapshot, executions)
            _report(symbol, cursor, reported_since, reported_until)
            if cursor.unattributed_steps != 0:
                _log_unattributed(symbol, cursor.unattributed_steps, read_since)

    def _clear(self, symbol: Symbol, cursor: _SymbolCursor) -> None:
        for tracking_id in self._ids_by_symbol[symbol]:
            self._exposure.pop(tracking_id, None)
        cursor.external_steps = 0
        cursor.external_opened_at = None

    def _fold(
        self,
        symbol: Symbol,
        cursor: _SymbolCursor,
        execution: Execution,
        signed_steps: StepCount,
    ) -> None:
        """A tag names the declared id whose order produced the fill, so the
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

        Folding in whole steps, the unit a fill is never finer than, keeps
        two fills that cancel exactly from leaving any residue behind.
        """
        before = cursor.net_steps
        after = before + signed_steps
        cursor.net_steps = after
        tracking_id = self._owner(symbol, execution)
        if tracking_id is not None:
            self._exposure[tracking_id] = (
                self._exposure.get(tracking_id, 0) + signed_steps
            )
            return
        held_outside = cursor.external_steps != 0
        if _passes_through_flat(before, after):
            self._clear(symbol, cursor)
            cursor.external_steps = after
        else:
            cursor.external_steps += signed_steps
        if not held_outside and cursor.external_steps == 0:
            return
        cursor.external_moved_at = execution.timestamp
        self._reporting.add(symbol)
        if cursor.external_steps == 0:
            cursor.external_opened_at = None
        elif cursor.external_opened_at is None:
            cursor.external_opened_at = execution.timestamp

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
        self._walk(symbol, cursor, _on_symbol(executions, symbol))
        self._settle(symbol, cursor, account_snapshot)
        return cursor

    def _refresh_symbol(
        self, symbol: Symbol, account_snapshot: AccountSnapshot
    ) -> None:
        cursor = self._cursors.setdefault(symbol, _SymbolCursor())
        self._walk(symbol, cursor, account_snapshot._declared_executions(symbol))
        self._settle(symbol, cursor, account_snapshot)

    def _settle(
        self, symbol: Symbol, cursor: _SymbolCursor, account_snapshot: AccountSnapshot
    ) -> None:
        position = account_snapshot.position(symbol)
        venue_net = 0.0 if position is None else signed_position(position)
        unattributed = to_step_count(venue_net) - cursor.net_steps
        if unattributed == cursor.unattributed_steps:
            return
        cursor.unattributed_steps = unattributed
        if unattributed == 0:
            self._unanchored.discard(symbol)
        else:
            self._unanchored.add(symbol)

    def _walk(
        self, symbol: Symbol, cursor: _SymbolCursor, executions: Iterable[Execution]
    ) -> None:
        new_executions = _new_since(cursor, executions)
        for execution in new_executions:
            self._fold(symbol, cursor, execution, _signed_steps(execution))
        if new_executions:
            _advance_cursor(cursor, new_executions)


def _require_every_id_tagged(
    ids_by_symbol: Mapping[Symbol, tuple[TrackingId, ...]],
    tags: Mapping[TrackingId, ProfileTag],
) -> None:
    declared_ids = {
        tracking_id for ids in ids_by_symbol.values() for tracking_id in ids
    }
    missing = sorted(declared_ids - tags.keys())
    if missing:
        raise StrategyCriticalError(
            f"Tracking id(s) {missing} have no tag; give every id declared in "
            "requirements.tracker.add(tags=...) its own tag."
        )


def _id_by_symbol_tag(
    ids_by_symbol: Mapping[Symbol, tuple[TrackingId, ...]],
    tags: Mapping[TrackingId, ProfileTag],
) -> dict[tuple[Symbol, str], TrackingId]:
    """A fill carries its symbol, so a tag only has to be unique among the
    ids declared on that symbol.
    """
    inverted: dict[tuple[Symbol, str], TrackingId] = {}
    for symbol, tracking_ids in ids_by_symbol.items():
        for tracking_id in tracking_ids:
            tag = tags[tracking_id]
            collision = inverted.get((symbol, tag))
            if collision is not None:
                raise StrategyCriticalError(
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
    """The engine carries nothing between cycles, so the candle a fill lands
    in is what makes the report of what it moved fire once: a fill booked
    after the close, before the cycle read it, belongs to the next candle.
    """
    moved_at = cursor.external_moved_at
    if moved_at is not None and reported_since <= moved_at < reported_until:
        _log_external(symbol, cursor)
    elif cursor.external_steps != 0:
        _log_standing_external(symbol, cursor)


def _log_external(symbol: Symbol, cursor: _SymbolCursor) -> None:
    """An external share and the moment it was opened stand or go together,
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


def _on_symbol(executions: Iterable[Execution], symbol: Symbol) -> list[Execution]:
    return [execution for execution in executions if execution.symbol == symbol]


def _new_since(
    cursor: _SymbolCursor, executions: Iterable[Execution]
) -> list[Execution]:
    """Two fills sharing an exact timestamp are common enough that a boundary
    of timestamp alone would silently drop whichever of them a later refresh
    had not yet folded, so the cursor also tracks which ids at that
    timestamp are already folded.
    """
    boundary = cursor.latest_timestamp
    new = [
        execution
        for execution in executions
        if boundary is None
        or execution.timestamp > boundary
        or (
            execution.timestamp == boundary
            and execution.execution_id not in cursor.seen_at_latest
        )
    ]
    return sorted(
        new, key=lambda execution: (execution.timestamp, execution.execution_id)
    )


def _signed_steps(execution: Execution) -> StepCount:
    steps = to_step_count(execution.quantity)
    return steps if execution.side == OrderSide.BUY else -steps


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
