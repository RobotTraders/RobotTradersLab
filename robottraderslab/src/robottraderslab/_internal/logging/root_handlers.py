import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager


@contextmanager
def handlers_on_root(handlers: Iterable[logging.Handler]) -> Iterator[None]:
    """Hold the handlers on the root logger for the block.

    Every handler is off the root logger again once the block ends, whatever
    the block raises.

    Args:
        handlers: The handlers to add to the root logger.
    """
    root = logging.getLogger()
    held = list(handlers)
    for handler in held:
        root.addHandler(handler)
    try:
        yield
    finally:
        for handler in held:
            root.removeHandler(handler)
