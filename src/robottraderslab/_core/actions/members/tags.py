from ...exceptions import BotCriticalError
from ...timeframes import TIMEFRAMES, TimeFrame

CUSTOM_TAG_BUDGET = 8

SEPARATOR = "-"

BRACKET_SUFFIXES: frozenset[str] = frozenset({"sl", "tp"})

FLATTEN_TAG = "flatten"

ENGINE_TAGS: frozenset[str] = BRACKET_SUFFIXES | {
    "sl-reached",
    "tp-reached",
    FLATTEN_TAG,
}


class ProfileTag(str):
    """A profile's whole identity on the wire, refused at construction unless
    every venue's client order id can carry it and no venue's readback takes
    its id for a bracket's.
    """

    __slots__ = ()

    def __new__(cls, value: str) -> "ProfileTag":
        """Hold the tag to what a client order id can carry.

        Args:
            value: The timeframe, followed by `-` and the custom part when the
                profile sets one.

        Raises:
            BotCriticalError: If the value does not open with a
                timeframe, or its custom part carries the separator, is `sl`
                or `tp`, or exceeds the byte budget a client order id allows it.
        """
        timeframe, separator, custom = value.partition(SEPARATOR)
        if timeframe not in TIMEFRAMES:
            raise BotCriticalError(
                f"`{timeframe}` is not a timeframe; a profile tag starts with one."
            )
        if separator:
            _require_carryable(custom)
        return super().__new__(cls, value)


def profile_tag(timeframe: TimeFrame, tag: str) -> ProfileTag:
    """Return the tag a profile's orders carry in their client order id.

    The timeframe keeps two profiles distinguishable on the wire when
    nothing but the timeframe sets them apart.

    Args:
        timeframe: The timeframe the profile trades on; it leads the tag even
            when `tag` is empty.
        tag: The profile's own label; an empty one leaves the timeframe alone.

    Raises:
        BotCriticalError: If the timeframe is unknown, the tag carries
            `-`, is `sl` or `tp`, or it exceeds the byte budget a client order id allows it.
    """
    if not tag:
        return ProfileTag(timeframe)
    return ProfileTag(f"{timeframe}{SEPARATOR}{tag}")


def require_not_a_bracket_suffix(tag: str) -> None:
    """Keep a custom tag from ending a client order id the way a bracket's does.

    Raises:
        ValueError: If the tag is `sl` or `tp`.
    """
    if tag in BRACKET_SUFFIXES:
        raise ValueError(
            f"'{tag}' ends a bracket's client order id on Kraken, so the id "
            "would read as the bracket of the entry before it; choose another"
        )


def require_within_budget(tag: str) -> None:
    """Hold a custom tag to the bytes every venue's client order id leaves it.

    Raises:
        ValueError: If the tag exceeds `CUSTOM_TAG_BUDGET` encoded bytes.
    """
    size = len(tag.encode())
    if size > CUSTOM_TAG_BUDGET:
        raise ValueError(
            f"'{tag}' is {size} bytes, over the {CUSTOM_TAG_BUDGET}-byte budget "
            "every venue's client order id honours"
        )


def engine_tag_refusal(tag: str) -> BotCriticalError:
    """Return the refusal for a tag the engine and the connectors write on the
    protections and closes they place, so an order carrying one is always
    theirs.
    """
    return BotCriticalError(
        f"tag `{tag}` is the engine's own; choose another for the order."
    )


def tag_of(client_order_id: str | None) -> str | None:
    """Return the tag a client order id carries, the whole label the order
    builder's `tag` set, such as `"1h-fast"`; None when it carries none.

    Args:
        client_order_id: Id the venue reports for an order, which may have
            been assigned by another process, another program, or by hand.
    """
    if client_order_id is None:
        return None
    _, _, after_process_tag = client_order_id.partition(SEPARATOR)
    _, separator, tag = after_process_tag.partition(SEPARATOR)
    return tag if separator and tag else None


def client_order_id_carrying(tag: str) -> str:
    """Return an id `tag_of` reads the tag back from.

    A venue too narrow for a whole id keeps the tag alone, so the segments
    before it come back empty.
    """
    return f"{SEPARATOR}{SEPARATOR}{tag}"


def profile_identity(symbol: str, timeframe: TimeFrame, tag: str) -> str:
    """Name the profile a trade belongs to, on charts and in reports.

    A profile is identified by the market it trades and the tag telling it
    apart from another on that market, so a report names it the same way
    whatever a strategy writes on its orders.
    """
    if not tag:
        return f"{symbol}@{timeframe}"
    return f"{symbol}@{timeframe}{SEPARATOR}{tag}"


def _require_carryable(custom: str) -> None:
    if SEPARATOR in custom:
        raise BotCriticalError(
            f"tag `{custom}` contains `{SEPARATOR}`, which separates the "
            "timeframe from the tag in the client order id."
        )
    try:
        require_not_a_bracket_suffix(custom)
        require_within_budget(custom)
    except ValueError as error:
        raise BotCriticalError(f"tag {error}.") from error
