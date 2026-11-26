from collections.abc import Iterable, Mapping
from itertools import chain

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
            ValueError: If the builder's `build()` refuses the order.
            BotCriticalError: If the order's symbol was never declared for the
                account.
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

    def _replace_actions(self, actions: Iterable[BaseExchangeAction]) -> None:
        """A wait declared on an action left out is dropped, since the executor
        never waits on an id it was not given.

        Args:
            actions: Every action to run, in the sequence to run them.
        """
        held = {action.id: action for action in actions}
        self._actions = held
        self._waits_for = {
            action_id: tuple(wait for wait in waits if wait in held)
            for action_id, waits in self._waits_for.items()
            if action_id in held
        }

    def _replace_with_followers(
        self,
        actions: Iterable[BaseExchangeAction],
        followers: Mapping[ActionID, ActionID],
    ) -> None:
        """Replace the actions, then order each follower after its leader.

        The actions are replaced as `_replace_actions` does. A follower runs
        after the action it follows and after what that action was booked to
        wait on, its own followers included, when it was booked to wait on
        anything, and every action booked to wait on the one followed waits
        for its followers as well.

        Args:
            actions: Every action to run, in the sequence to run them.
            followers: Each action of `actions` that stands for part of the work
                of another, by the id of that other action.
        """
        self._replace_actions(actions)
        self._waits_for = _with_followers(self._waits_for, followers)


def _ids_of(
    after: BaseExchangeAction | Iterable[BaseExchangeAction],
) -> tuple[ActionID, ...]:
    if isinstance(after, BaseExchangeAction):
        return (after.id,)
    return tuple(action.id for action in after)


def _with_followers(
    waits: dict[ActionID, tuple[ActionID, ...]],
    followers: Mapping[ActionID, ActionID],
) -> dict[ActionID, tuple[ActionID, ...]]:
    followed: dict[ActionID, list[ActionID]] = {}
    for follower, leader in followers.items():
        followed.setdefault(leader, []).append(follower)
    with_followers = {
        action_id: (*awaited, *_followers_of(awaited, followed))
        for action_id, awaited in waits.items()
    }
    for follower, leader in followers.items():
        if leader in waits:
            with_followers[follower] = (*with_followers[leader], leader)
    return with_followers


def _followers_of(
    awaited: Iterable[ActionID], followed: Mapping[ActionID, list[ActionID]]
) -> tuple[ActionID, ...]:
    return tuple(chain.from_iterable(followed.get(wait, ()) for wait in awaited))
