from collections.abc import Iterable

from .exceptions import ActionAlreadyExistsError, ActionNotFoundError
from .members import ActionBuilder, ActionID, BaseExchangeAction, finished


class BookKeeper:
    """Collects the actions a strategy books, and the order they must run in."""

    def __init__(self) -> None:
        self._actions: dict[ActionID, BaseExchangeAction] = {}
        self._waits_for: dict[ActionID, tuple[ActionID, ...]] = {}

    def add(
        self,
        action: BaseExchangeAction | ActionBuilder[BaseExchangeAction],
        after: BaseExchangeAction | Iterable[BaseExchangeAction] | None = None,
    ) -> BaseExchangeAction:
        """
        Hand an action to the engine, to run after the candle is booked.

        Args:
            action: A finished action, or an order builder, which the
                bookkeeper finishes with its `build()`.
            after: Actions this one must follow, each already booked. Omitted,
                it follows every action booked before it; an empty sequence,
                none.

        Returns:
            The booked action, to name in a later action's `after`.

        Raises:
            ActionAlreadyExistsError: If the action is already booked.
            ActionNotFoundError: If `after` names an action that is not booked.
            StrategyCriticalError: If the builder's `build()` refuses the order.
        """
        action = finished(action)
        if action.id in self._actions:
            raise ActionAlreadyExistsError(action.id)
        if after is not None:
            waits_for = _ids_of(after)
            for action_id in waits_for:
                if action_id not in self._actions:
                    raise ActionNotFoundError(action_id)
            self._waits_for[action.id] = waits_for
        self._actions[action.id] = action
        return action

    def declared_waits(self) -> dict[ActionID, tuple[ActionID, ...]]:
        """Return what each action must follow, as declared with `after`.

        Returns:
            The ids to follow, per action that declared any. An action absent
            from the mapping follows every action booked before it.
        """
        return dict(self._waits_for)

    def list_actions(self) -> list[BaseExchangeAction]:
        """Return every booked action, in booking order."""
        return list(self._actions.values())


def _ids_of(
    after: BaseExchangeAction | Iterable[BaseExchangeAction],
) -> tuple[ActionID, ...]:
    if isinstance(after, BaseExchangeAction):
        return (after.id,)
    return tuple(action.id for action in after)
