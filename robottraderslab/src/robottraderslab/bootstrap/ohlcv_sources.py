CSV = "csv"
MOCK = "mock"
SOURCES_READING_NO_VENUE = (CSV, MOCK)


def keeps_candles(source: str) -> bool:
    """A source reading a venue keeps what it downloads, and one reading a file
    of its own or generating its candles has nothing to keep, so only the first
    is given a store.

    Args:
        source: The name a config gives under `ohlcv_provider`.
    """
    return source.lower() not in SOURCES_READING_NO_VENUE
