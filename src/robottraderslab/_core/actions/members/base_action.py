from dataclasses import dataclass, field
from itertools import count
from uuid import uuid4

from ...symbol import Symbol
from .action_result import ActionResult
from .tags import SEPARATOR

type ActionID = int

_PROCESS_TAG_LENGTH = 12

_next_action_id = count(1).__next__
_process_tag = uuid4().hex[:_PROCESS_TAG_LENGTH]


@dataclass(kw_only=True, slots=True)
class BaseExchangeAction:
    """One thing a strategy books for the engine to carry out on a venue.

    Attributes:
        tag: A label kept in the client order id, so a later run recognises
            the resulting order as its own and reads it back with `tag_of`;
            None for an unlabelled order, whose client order id carries no
            tag.
    """

    id: ActionID = field(init=False, default_factory=_next_action_id)
    symbol: Symbol
    tag: str | None = None

    @property
    def client_order_id(self) -> str:
        if self.tag is None:
            return f"{_process_tag}{SEPARATOR}{self.id}"
        return f"{_process_tag}{SEPARATOR}{self.id}{SEPARATOR}{self.tag}"

    async def execute(self) -> ActionResult:
        """Carry the action out on the venue.

        Returns:
            What the venue accepted, empty for an action that places no order.

        Raises:
            ExchangeRecoverableError: If the venue rejected the action.
            ExchangeTransientError: If the attempt is worth retrying.
            ExchangeCriticalError: If the venue is unusable.
        """
        raise NotImplementedError
