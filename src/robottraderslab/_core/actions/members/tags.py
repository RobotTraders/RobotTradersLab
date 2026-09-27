from ...exceptions import StrategyCriticalError
from ...timeframes import TIMEFRAMES, TimeFrame

CUSTOM_TAG_BUDGET = 8

SEPARATOR = "-"


class ProfileTag(str):
    """A profile's whole identity on the wire, `timeframe` or `timeframe-tag`,
    refused at construction unless every venue's client order id can carry it.
    """

    __slots__ = ()

    def __new__(cls, value: str) -> "ProfileTag":
        """Build the tag, holding it to what a client order id can carry.

        Raises:
            StrategyCriticalError: If the value does not open with a
                timeframe, or its custom part carries the separator or
                exceeds `CUSTOM_TAG_BUDGET` encoded bytes.
        """
        timeframe, separator, custom = value.partition(SEPARATOR)
        if timeframe not in TIMEFRAMES:
            raise StrategyCriticalError(
                f"`{timeframe}` is not a timeframe; a profile tag starts with one."
            )
        if separator:
            _require_carryable(custom)
        return super().__new__(cls, value)


def profile_tag(timeframe: TimeFrame, tag: str) -> ProfileTag:
    """Return the tag a profile's orders carry in their client order id, the
    timeframe followed by `-` and the tag when one is given.

    The timeframe keeps two profiles distinguishable on the wire when
    nothing but the timeframe sets them apart.

    Raises:
        StrategyCriticalError: If the timeframe is unknown, the tag carries
            `-`, or it exceeds 8 encoded bytes.
    """
    if not tag:
        return ProfileTag(timeframe)
    return ProfileTag(f"{timeframe}{SEPARATOR}{tag}")


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
        raise StrategyCriticalError(
            f"tag `{custom}` contains `{SEPARATOR}`, which separates the "
            "timeframe from the tag in the client order id."
        )
    try:
        require_within_budget(custom)
    except ValueError as error:
        raise StrategyCriticalError(f"tag {error}.") from error
