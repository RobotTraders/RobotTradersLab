import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence

from ..exceptions import ExchangeRecoverableError, ExchangeTransientError
from ..notifications import OnOrderFilled, OnOrderPlaced
from ..order import (
    OnFillRead,
    OrderFill,
    OrderPlacement,
    PlacedOrder,
    VenueFill,
)
from ..retry import (
    BASE_DELAY_SECONDS,
    MAX_ATTEMPTS,
    retry_on_transient,
)
from .members import ActionID, ActionResult, BaseExchangeAction

logger = logging.getLogger(__name__)


async def execute_trading_actions(
    actions: Sequence[BaseExchangeAction],
    on_order_filled: Iterable[OnOrderFilled] = (),
    *,
    on_order_placed: Iterable[OnOrderPlaced] = (),
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY_SECONDS,
    declared_waits: Mapping[ActionID, tuple[ActionID, ...]] | None = None,
) -> list[Callable[[], Awaitable[None]]]:
    """Each action starts once the actions it must follow have finished, so
    actions that do not follow each other run at the same time. An action
    that declared nothing follows every action booked before it, which is
    one action at a time.

    Args:
        on_order_placed: Callbacks to fire for each order the venue accepted.
        max_attempts: Total retry attempts for transient errors.
        base_delay: Base delay in seconds (doubled on each retry).
        declared_waits: What each action waits for, as booked with `after`.
            Actions missing from it wait for everything booked before them.

    Returns:
        Async callbacks to fire after all actions are executed.

    Raises:
        ExchangeCriticalError: If the venue is unusable.
        StrategyCriticalError: If an action names a symbol the venue cannot
            trade.
    """
    execution_callbacks = list(on_order_filled)
    placement_callbacks = list(on_order_placed)
    if not declared_waits:
        return await _execute_in_order(
            actions, execution_callbacks, placement_callbacks, max_attempts, base_delay
        )
    return await _execute_when_ready(
        actions,
        declared_waits,
        execution_callbacks,
        placement_callbacks,
        max_attempts,
        base_delay,
    )


async def execute_setting(
    action: BaseExchangeAction,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY_SECONDS,
) -> bool:
    """Execute an action that places no order.

    Args:
        max_attempts: Total retry attempts for transient errors.
        base_delay: Base delay in seconds (doubled on each retry).

    Returns:
        Whether the venue carried it out; a rejection or a failure is logged.

    Raises:
        ExchangeCriticalError: If the venue is unusable.
        StrategyCriticalError: If the action names a symbol the venue cannot
            trade.
    """
    return await _attempt(action, max_attempts, base_delay) is not None


async def _execute_in_order(
    actions: Sequence[BaseExchangeAction],
    execution_callbacks: list[OnOrderFilled],
    placement_callbacks: list[OnOrderPlaced],
    max_attempts: int,
    base_delay: float,
) -> list[Callable[[], Awaitable[None]]]:
    post_execution_callbacks: list[Callable[[], Awaitable[None]]] = []
    for action in actions:
        post_execution_callbacks.extend(
            await _execute_one(
                action,
                execution_callbacks,
                placement_callbacks,
                max_attempts,
                base_delay,
            )
        )
    return post_execution_callbacks


async def _execute_when_ready(
    actions: Sequence[BaseExchangeAction],
    declared_waits: Mapping[ActionID, tuple[ActionID, ...]],
    execution_callbacks: list[OnOrderFilled],
    placement_callbacks: list[OnOrderPlaced],
    max_attempts: int,
    base_delay: float,
) -> list[Callable[[], Awaitable[None]]]:
    started: dict[ActionID, asyncio.Task[list[Callable[[], Awaitable[None]]]]] = {}

    async def run_when_ready(
        action: BaseExchangeAction, waits_for: tuple[ActionID, ...]
    ) -> list[Callable[[], Awaitable[None]]]:
        for dependency in waits_for:
            await started[dependency]
        return await _execute_one(
            action,
            execution_callbacks,
            placement_callbacks,
            max_attempts,
            base_delay,
        )

    for action in actions:
        waits_for = declared_waits.get(action.id, tuple(started))
        started[action.id] = asyncio.create_task(run_when_ready(action, waits_for))

    outcomes = await asyncio.gather(*started.values(), return_exceptions=True)
    post_execution_callbacks: list[Callable[[], Awaitable[None]]] = []
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
        post_execution_callbacks.extend(outcome)
    return post_execution_callbacks


async def _execute_one(
    action: BaseExchangeAction,
    execution_callbacks: list[OnOrderFilled],
    placement_callbacks: list[OnOrderPlaced],
    max_attempts: int,
    base_delay: float,
) -> list[Callable[[], Awaitable[None]]]:
    result = await _attempt(action, max_attempts, base_delay)
    if result is None:
        return []
    return [
        _create_post_execution_callback(
            outcome.placed_order,
            outcome.on_filled,
            outcome.placement,
            execution_callbacks,
            placement_callbacks,
        )
        for outcome in result.orders
    ]


async def _attempt(
    action: BaseExchangeAction, max_attempts: int, base_delay: float
) -> ActionResult | None:
    try:
        return await retry_on_transient(
            action.execute, max_attempts=max_attempts, base_delay=base_delay
        )
    except ExchangeTransientError as e:
        logger.error(
            f"{type(action).__name__} ({action.symbol}) skipped after "
            f"{max_attempts} attempts: {e}"
        )
    except ExchangeRecoverableError as e:
        logger.warning(
            f"Exchange rejected {type(action).__name__} ({action.symbol}): {e}"
        )
    except Exception as e:
        logger.error(
            f"Internal error in {type(action).__name__} ({action.symbol}): {e}"
        )
    return None


def _as_order_fill(fill: VenueFill, placement: OrderPlacement) -> OrderFill:
    """Only a strategy books an action, so only a strategy fires the fills of one.

    The effect stays unstated: an action carries no account, so the position
    the fill acted on is out of reach here.
    """
    return OrderFill(
        order_id=fill.order_id,
        symbol=fill.symbol,
        side=fill.side,
        quantity=fill.quantity,
        kind=placement.kind,
        timestamp=fill.timestamp,
        filled_value=fill.filled_value,
        client_order_id=fill.client_order_id or placement.client_order_id,
        reason=placement.reason,
        source="strategy",
    )


def _create_post_execution_callback(
    placed_order: PlacedOrder,
    on_filled: OnFillRead | None,
    placement: OrderPlacement,
    execution_callbacks: list[OnOrderFilled],
    placement_callbacks: list[OnOrderPlaced],
) -> Callable[[], Awaitable[None]]:
    async def _callback() -> None:
        for placement_callback in placement_callbacks:
            try:
                await placement_callback(placement)
            except Exception as e:
                logger.error(
                    "Placement callback %s failed: %s",
                    type(placement_callback).__qualname__,
                    e,
                )
        fill = await placed_order.get_fill()
        order_fill = _as_order_fill(fill, placement) if fill is not None else None
        if order_fill is not None:
            logger.info("Order filled | %s", order_fill)
        else:
            logger.info("Order placed | %s", placement)
        if on_filled is not None:
            await on_filled(placed_order, order_fill)
        if order_fill is not None:
            for callback in execution_callbacks:
                try:
                    await callback(order_fill)
                except Exception as e:
                    logger.error(
                        "Execution callback %s failed: %s",
                        type(callback).__qualname__,
                        e,
                    )

    return _callback
