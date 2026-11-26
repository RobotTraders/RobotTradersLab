import logging
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

from robottraderslab._internal import (
    booked_items,
    closed_steps,
    is_immediate,
    is_sent_at_once,
    rebuilt_actions,
    without_protections,
)
from robottraderslab.futures import (
    BatchableOrderAction,
    CancelOrderByIdAction,
    CancelOrdersForSymbolAction,
    ClosePositionAction,
    FuturesLimitOrderAction,
    FuturesMarketOrderAction,
    FuturesOrderBatchAction,
    FuturesOrderModifyAction,
    SetLeverageAction,
    SetMarginModeAction,
    UpdatePositionStopLossAction,
    UpdatePositionTakeProfitAction,
    UpdatePositionTrailingStopAction,
)
from robottraderslab.strategies import AccountSnapshots
from robottraderslab_core import Symbol
from robottraderslab_core.engine import (
    BRACKET_SUFFIXES,
    PROTECTION_KINDS,
    RESTING_ENTRY_KINDS,
    AccountRequirement,
    ActionID,
    StepCount,
    carries_this_bot,
    position_steps,
    to_quantity,
    to_step_count,
)
from robottraderslab_core.exchanges import (
    BaseExchangeAction,
    OrderProtocol,
    OrderSide,
    OrderType,
    PositionSnapshot,
    tag_of,
)
from robottraderslab_core.strategies import AccountRequirements

logger = logging.getLogger(__name__)

_WHOLE_POSITION = 1.0
_KEPT_AS_BOOKED: tuple[type[BaseExchangeAction], ...] = (
    UpdatePositionStopLossAction,
    UpdatePositionTakeProfitAction,
    UpdatePositionTrailingStopAction,
    SetLeverageAction,
    SetMarginModeAction,
)
_JUDGED: tuple[type[BaseExchangeAction], ...] = (
    ClosePositionAction,
    FuturesLimitOrderAction,
    FuturesMarketOrderAction,
    FuturesOrderBatchAction,
    FuturesOrderModifyAction,
    CancelOrderByIdAction,
    CancelOrdersForSymbolAction,
)


@dataclass(frozen=True, slots=True)
class CutOrder:
    """An order sent for less than it was booked for."""

    original: BatchableOrderAction
    quantity: float


@dataclass(frozen=True, slots=True)
class ExitOnlyDecision:
    """The actions are in booking order. An order cut or stripped of its
    protections is a copy carrying the id of the order it stands for, and a
    batch or a modification keeps only the members let through. A resting
    order is cancelled when it could open, increase or flip the position and
    is not kept.
    """

    actions: tuple[BaseExchangeAction, ...]
    cut: tuple[CutOrder, ...]
    dropped: tuple[BaseExchangeAction, ...]
    cancelled: tuple[OrderProtocol, ...]


def restrict_to_exit_only(
    requirements: AccountRequirements,
    account_snapshots: AccountSnapshots,
    exit_only_symbols: Collection[Symbol],
    actions: Sequence[BaseExchangeAction],
) -> list[BaseExchangeAction]:
    """Book the cancels of the resting orders that could open or increase a
    position, ahead of the actions let through.

    Args:
        requirements: Account declarations collected in `setup`.
        account_snapshots: The snapshots read for this candle, which hold the
            positions and open orders of every account declaring an exit-only
            symbol.
        exit_only_symbols: The symbols whose booked actions are restricted.
        actions: Everything the strategy booked this candle, in booking order.
    """
    cancels: list[BaseExchangeAction] = []
    judged: set[Symbol] = set()
    for requirement in requirements._get_all():
        symbols = [
            symbol
            for symbol in requirement.symbols
            if symbol in exit_only_symbols and symbol not in judged
        ]
        if not symbols:
            continue
        actions = restrict_account(
            requirement, account_snapshots, symbols, actions, cancels
        )
        judged.update(symbols)
    return [*cancels, *actions]


def restrict_account(
    requirement: AccountRequirement,
    account_snapshots: AccountSnapshots,
    symbols: Iterable[Symbol],
    actions: Sequence[BaseExchangeAction],
    cancels: list[BaseExchangeAction],
) -> list[BaseExchangeAction]:
    """Judge each symbol against the account's own position and open orders.

    Each cut, dropped or cancelled order is logged.

    Args:
        requirement: What the strategy declared for the account.
        account_snapshots: The snapshots read for this candle, which hold the
            account's positions and open orders.
        symbols: The account's symbols run in exit-only mode.
        actions: Everything booked this candle, in booking order.
        cancels: Receives the cancels of the resting orders the mode books.

    Returns:
        The actions let through, in booking order.
    """
    snapshot = account_snapshots.of(requirement.account)
    labels = requirement.account._labels_resting_triggers()
    for symbol in sorted(symbols):
        decision = restrict_symbol(
            symbol,
            actions,
            snapshot.position(symbol),
            snapshot.open_orders(symbol),
            labels_resting_triggers=labels,
        )
        _log(symbol, decision)
        actions = decision.actions
        cancels.extend(
            requirement.account.cancel_order(symbol, order.order_id)
            for order in decision.cancelled
        )
    return list(actions)


def restrict_symbol(
    symbol: Symbol,
    actions: Sequence[BaseExchangeAction],
    position: PositionSnapshot | None,
    open_orders: Sequence[OrderProtocol],
    *,
    labels_resting_triggers: bool,
) -> ExitOnlyDecision:
    """No order sent opens, increases or flips the venue's position, since each
    is counted against what the position can still absorb. What remains of the
    position is counted in this sequence: the market orders and closes, the
    resting orders not cancelled, the limit and trigger orders booked.

    A close of a share of the position can close the whole of it, since the
    venue rounds the share to its lot and closes the whole position when the
    remainder would be worth less than its minimum order value. After such a
    close the position absorbs nothing more, and the exposure counts as not
    ended unless the orders booked ahead of it end it.

    The resting orders concerned are the bot's own limit and trigger orders
    that are not flagged reduce-only, which can open, increase or flip the
    position. One the venue reports with no flag is judged by its side and
    size like any other. One on the position's side, or whose side or size the
    venue does not state, is cancelled, and so is every one on a symbol
    holding no position. One on the closing side is kept while what remains of
    the position holds its size, the first listed first.

    Args:
        symbol: The symbol in exit-only mode; an action on another symbol is
            returned as booked.
        actions: Everything the strategy booked this candle, in booking order.
        position: The venue's position on the symbol, None when it holds none.
        open_orders: The resting orders on the symbol, oldest first, a
            protection and a trigger entry among them.
        labels_resting_triggers: Whether the venue keeps the caller's client
            order id on a resting trigger. Without it, a trigger carries no id
            to tell it from another's, so every resting trigger counts as the
            bot's own.

    Returns:
        A call with the same arguments gives the same decision.
    """
    ledger = _Ledger(symbol, position)
    booked = [item for _, item in booked_items(actions) if item.symbol == symbol]
    ledger.settle_market_orders(booked)
    by_id = {order.order_id: order for order in open_orders}
    covered = _judge_cancels(ledger, actions, by_id)
    candidates = [
        order
        for order in open_orders
        if _is_resting_candidate(order, labels_resting_triggers)
        and order.order_id not in covered
    ]
    cancelled = ledger.settle_resting_orders(candidates)
    ledger.settle_non_immediate_orders(booked)
    _judge_modifications(
        ledger, actions, by_id, {order.order_id for order in cancelled}
    )
    ledger.drop_unknown(actions)
    return ledger.decision(actions, cancelled)


class _Ledger:
    def __init__(self, symbol: Symbol, position: PositionSnapshot | None) -> None:
        steps = position_steps(position)
        self._symbol = symbol
        self._flat = steps == 0
        self._closing_side: OrderSide | None = (
            None if position is None or self._flat else position.side.closing_side
        )
        self._ends_exposure = self._flat
        self._may_close_whole = False
        self._remaining: StepCount = abs(steps)
        self._replaced: dict[ActionID, BatchableOrderAction] = {}
        self._cut: list[CutOrder] = []
        self._dropped: list[BaseExchangeAction] = []

    @property
    def closing_side(self) -> OrderSide | None:
        return self._closing_side

    @property
    def ends_exposure(self) -> bool:
        return self._ends_exposure

    @property
    def flat(self) -> bool:
        return self._flat

    @property
    def symbol(self) -> Symbol:
        return self._symbol

    @property
    def _room(self) -> StepCount:
        return 0 if self._may_close_whole else self._remaining

    def decision(
        self,
        actions: Sequence[BaseExchangeAction],
        cancelled: Sequence[OrderProtocol],
    ) -> ExitOnlyDecision:
        gone = {action.id for action in self._dropped}
        return ExitOnlyDecision(
            actions=tuple(rebuilt_actions(actions, gone, self._replaced)),
            cut=tuple(self._cut),
            dropped=tuple(self._dropped),
            cancelled=tuple(cancelled),
        )

    def drop(self, action: BaseExchangeAction) -> None:
        self._dropped.append(action)

    def drop_unknown(self, actions: Sequence[BaseExchangeAction]) -> None:
        for action in actions:
            if action.symbol == self._symbol and not isinstance(
                action, (*_KEPT_AS_BOOKED, *_JUDGED)
            ):
                self.drop(action)

    def keep_modification(self, order: BatchableOrderAction, reducing: bool) -> None:
        if not reducing:
            self._strip_protections(order)

    def settle_market_orders(
        self, items: Sequence[BatchableOrderAction | ClosePositionAction]
    ) -> None:
        self._settle(items, immediate=True)
        self._ends_exposure = self._remaining == 0

    def settle_non_immediate_orders(
        self, items: Sequence[BatchableOrderAction | ClosePositionAction]
    ) -> None:
        self._settle(items, immediate=False)

    def settle_resting_orders(
        self, orders: Sequence[OrderProtocol]
    ) -> list[OrderProtocol]:
        cancelled: list[OrderProtocol] = []
        for order in orders:
            steps = _resting_steps(order)
            if (
                self._closing_side is None
                or order.side is not self._closing_side
                or steps is None
                or steps > self._room
            ):
                cancelled.append(order)
            else:
                self._remaining -= steps
        return cancelled

    def _close(self, close: ClosePositionAction) -> None:
        self._remaining -= closed_steps(self._remaining, close.closing_ratio)
        if close.closing_ratio < _WHOLE_POSITION:
            self._may_close_whole = True

    def _cut_to(self, order: BatchableOrderAction, steps: StepCount) -> None:
        quantity = to_quantity(steps)
        cut = without_protections(order)
        cut.quantity = quantity
        self._replaced[order.id] = cut
        self._cut.append(CutOrder(order, quantity))

    def _place(self, order: BatchableOrderAction) -> None:
        steps = to_step_count(order.quantity)
        reduces = order.side is self._closing_side
        room = self._room
        if order.reduce_only:
            if is_immediate(order) and reduces:
                self._remaining -= min(steps, self._remaining)
        elif not reduces or room == 0:
            self.drop(order)
        elif steps <= room:
            self._remaining -= steps
            self._strip_protections(order)
        else:
            self._cut_to(order, room)
            self._remaining = 0

    def _settle(
        self,
        items: Sequence[BatchableOrderAction | ClosePositionAction],
        *,
        immediate: bool,
    ) -> None:
        for item in items:
            if is_sent_at_once(item) is not immediate:
                continue
            if isinstance(item, ClosePositionAction):
                self._close(item)
            else:
                self._place(item)

    def _strip_protections(self, order: BatchableOrderAction) -> None:
        if _is_protected(order):
            self._replaced[order.id] = without_protections(order)


def _is_protected(order: BatchableOrderAction) -> bool:
    return order.stop_loss is not None or order.take_profit is not None


def _judge_cancels(
    ledger: _Ledger,
    actions: Sequence[BaseExchangeAction],
    by_id: dict[str, OrderProtocol],
) -> set[str]:
    covered: set[str] = set()
    for action in actions:
        if action.symbol != ledger.symbol:
            continue
        if isinstance(action, CancelOrderByIdAction):
            order = by_id.get(action.order_id)
            protection = order is None or order.kind in PROTECTION_KINDS
            if protection and not ledger.ends_exposure:
                ledger.drop(action)
            else:
                covered.add(action.order_id)
        elif isinstance(action, CancelOrdersForSymbolAction):
            if _may_cancel_orders(ledger, action.order_tag):
                covered.update(
                    order.order_id
                    for order in by_id.values()
                    if action.order_tag is None
                    or tag_of(order.client_order_id) == action.order_tag
                )
            else:
                ledger.drop(action)
    return covered


def _may_cancel_orders(ledger: _Ledger, tag: str | None) -> bool:
    if tag is None:
        return ledger.flat
    return tag not in BRACKET_SUFFIXES or ledger.ends_exposure


def _is_resting_candidate(order: OrderProtocol, labels_resting_triggers: bool) -> bool:
    return (
        order.kind in RESTING_ENTRY_KINDS
        and order.reduce_only is not True
        and (
            carries_this_bot(order.client_order_id)
            or (order.kind is OrderType.TRIGGER and not labels_resting_triggers)
        )
    )


def _resting_steps(order: OrderProtocol) -> StepCount | None:
    quantity = order.remaining_quantity
    return None if quantity is None else to_step_count(quantity)


def _judge_modifications(
    ledger: _Ledger,
    actions: Sequence[BaseExchangeAction],
    by_id: dict[str, OrderProtocol],
    cancelled_ids: set[str],
) -> None:
    for action in actions:
        if not isinstance(action, FuturesOrderModifyAction):
            continue
        for modification in action.modifications:
            order = modification.order
            if order.symbol != ledger.symbol:
                continue
            resting = by_id.get(modification.order_id)
            if (
                resting is not None
                and resting.order_id not in cancelled_ids
                and _only_reduces(ledger, resting, order)
            ):
                ledger.keep_modification(order, resting.reduce_only is True)
            else:
                ledger.drop(order)


def _only_reduces(
    ledger: _Ledger, resting: OrderProtocol, replacement: BatchableOrderAction
) -> bool:
    if resting.side is not ledger.closing_side or resting.quantity is None:
        return False
    return to_step_count(replacement.quantity) <= to_step_count(resting.quantity)


def _log(symbol: Symbol, decision: ExitOnlyDecision) -> None:
    for cut in decision.cut:
        logger.warning(
            "%s: exit-only mode cuts %s to %g",
            symbol,
            _describe(cut.original),
            cut.quantity,
        )
    for action in decision.dropped:
        logger.info("%s: exit-only mode drops %s", symbol, _describe(action))
    for order in decision.cancelled:
        logger.info(
            "%s: exit-only mode cancels the resting %s order %s",
            symbol,
            order.kind,
            order.order_id,
        )


def _describe(action: BaseExchangeAction) -> str:
    name = type(action).__name__
    if isinstance(action, (FuturesLimitOrderAction, FuturesMarketOrderAction)):
        return f"{name} ({action.side} {action.quantity:g})"
    if isinstance(action, CancelOrderByIdAction):
        return f"{name} ({action.order_id})"
    return name
