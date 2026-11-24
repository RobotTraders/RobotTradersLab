import logging
from collections.abc import Iterator

import pytest

from robottraderslab_core.engine import EXECUTOR_LOGGER_NAME


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    ccxt = logging.getLogger("ccxt")
    executor = logging.getLogger(EXECUTOR_LOGGER_NAME)
    handlers, root_level = root.handlers[:], root.level
    ccxt_level, executor_level = ccxt.level, executor.level
    yield
    root.handlers = handlers
    root.setLevel(root_level)
    ccxt.setLevel(ccxt_level)
    executor.setLevel(executor_level)
