from collections.abc import Iterable

import pytest

from robottraderslab import Symbol
from robottraderslab._internal import DryRunFailure


def logged(caplog: pytest.LogCaptureFixture) -> list[tuple[int, str]]:
    """Each record the log held, as its level number and its message."""
    return [(record.levelno, record.getMessage()) for record in caplog.records]


def outcomes(failures: Iterable[DryRunFailure]) -> list[tuple[Symbol | None, bool]]:
    """Each failure's symbol and whether it is recoverable."""
    return [(failure.symbol, failure.recoverable) for failure in failures]
