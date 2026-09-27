from .base_action import BaseExchangeAction


# A plain base rather than an ABC or a Protocol: `BookKeeper.add` tests every
# action against this class, and a plain class keeps that test a type check.
class ActionBuilder[ActionT: BaseExchangeAction]:
    """An action still being put together, which the bookkeeper finishes with
    `build()` when it is handed one.
    """

    def build(self) -> ActionT:
        """Return the finished action.

        Raises:
            StrategyCriticalError: If what was put together is not an action
                the engine can carry out.
        """
        raise NotImplementedError


def finished[ActionT: BaseExchangeAction](
    action: ActionT | ActionBuilder[ActionT],
) -> ActionT:
    if isinstance(action, ActionBuilder):
        return action.build()
    return action
