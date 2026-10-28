from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import blake2b
from itertools import count
from string import ascii_lowercase, digits

from ...symbol import Symbol
from .action_result import ActionResult
from .tags import SEPARATOR

type ActionID = int

_BASE36: str = digits + ascii_lowercase
_SEGMENT_WIDTH: int = 6
_SEGMENT_SPAN: int = len(_BASE36) ** _SEGMENT_WIDTH
_BOT_DIGEST_SIZE: int = 8
_BOT_LEADING_BYTES: int = 2
_TEXT_SEGMENT_WIDTH: int = 2 * _SEGMENT_WIDTH
_LAUNCH_EPOCH: datetime = datetime(2026, 1, 1, tzinfo=UTC)

_next_action_id = count(1).__next__


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
            return f"{_order_id_prefix}{self.id}"
        return f"{_order_id_prefix}{self.id}{SEPARATOR}{self.tag}"

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


def next_client_order_id(tag: str) -> str:
    """The id of an order a connector places on its own account, drawn from the
    same counter as the strategy's orders so no two orders of one process share one.

    Args:
        tag: One of the engine's own tags in `ENGINE_TAGS`, such as
            `tp-reached`.
    """
    return f"{_order_id_prefix}{_next_action_id()}{SEPARATOR}{tag}"


def name_process(bot_name: str, launched_at: datetime) -> None:
    """A run calls it before it reads its first client order id, since each id
    takes the prefix current when read.

    A later process recognises the bot's orders as its own because the bot
    segment derives from the name alone. Two processes of one bot share a
    launch segment only when both start within the same second, which a config
    running one instance at a time leaves to a run lasting less than a second.

    Args:
        bot_name: The bot's name, its folder's.
        launched_at: The moment the run starts.
    """
    global _order_id_prefix, _process_bot_bytes, _process_bot_segment
    _order_id_prefix = _order_id_prefix_of(bot_name, launched_at)
    _process_bot_bytes = bot_leading_bytes(bot_name)
    _process_bot_segment = bot_segment(bot_name)


def process_bot_bytes() -> bytes:
    """The two leading bytes of the bot digest the process's bot segment derives
    from, for a venue whose order id holds no room for the whole segment.
    """
    return _process_bot_bytes


def carries_this_bot(client_order_id: str | None) -> bool:
    """A text venue reports the whole first segment, `{bot}{launch}`, and
    Hyperliquid the four hex characters of the bot's two leading bytes. An id
    placed before the bot was named opens with twelve hex characters no digest
    of a name produces, so it is never the bot's.

    Args:
        client_order_id: The id a venue reports for an order or a fill, placed
            by this bot, another bot or a hand.

    Returns:
        Whether the id's first segment names the bot this process runs.
    """
    if client_order_id is None:
        return False
    first = client_order_id.partition(SEPARATOR)[0]
    if len(first) == _TEXT_SEGMENT_WIDTH:
        return first[:_SEGMENT_WIDTH] == _process_bot_segment
    return first == _process_bot_bytes.hex()


def bot_segment(bot_name: str) -> str:
    """The segment is the same in every process, since orders are recognised
    across launches by it.
    """
    return _base36(int.from_bytes(_bot_digest(bot_name)) % _SEGMENT_SPAN)


def bot_leading_bytes(bot_name: str) -> bytes:
    """A venue whose order id has no room for the whole segment identifies the
    bot by these bytes.

    Args:
        bot_name: The bot's name, its folder's.
    """
    return _bot_digest(bot_name)[:_BOT_LEADING_BYTES]


def _bot_digest(bot_name: str) -> bytes:
    return blake2b(bot_name.encode(), digest_size=_BOT_DIGEST_SIZE).digest()


def _order_id_prefix_of(bot_name: str, launched_at: datetime) -> str:
    launch_seconds = int((launched_at - _LAUNCH_EPOCH).total_seconds())
    return f"{bot_segment(bot_name)}{_base36(launch_seconds)}{SEPARATOR}"


def _base36(number: int) -> str:
    characters: list[str] = []
    for _ in range(_SEGMENT_WIDTH):
        number, remainder = divmod(number, len(_BASE36))
        characters.append(_BASE36[remainder])
    return "".join(reversed(characters))


_order_id_prefix: str = _order_id_prefix_of("", datetime.now(UTC))
_process_bot_bytes: bytes = bot_leading_bytes("")
_process_bot_segment: str = bot_segment("")
