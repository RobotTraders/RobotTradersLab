from datetime import UTC, datetime

import pytest

from robottraderslab._core import candle_closes_at


@pytest.mark.parametrize(
    ("timeframe", "moment", "closes"),
    [
        ("1m", datetime(2026, 1, 1, 9, 15, tzinfo=UTC), True),
        ("1m", datetime(2026, 1, 1, 9, 16, 42, tzinfo=UTC), True),
        ("5m", datetime(2026, 1, 1, 9, 5, 2, tzinfo=UTC), True),
        ("5m", datetime(2026, 1, 1, 9, 6, tzinfo=UTC), False),
        ("15m", datetime(2026, 1, 1, 9, 15, tzinfo=UTC), True),
        ("15m", datetime(2026, 1, 1, 9, 14, 59, tzinfo=UTC), False),
        ("15m", datetime(2026, 1, 1, 9, 16, tzinfo=UTC), False),
        ("1h", datetime(2026, 1, 1, 10, 0, 30, tzinfo=UTC), True),
        ("1h", datetime(2026, 1, 1, 10, 30, tzinfo=UTC), False),
        ("4h", datetime(2026, 1, 1, 8, 0, tzinfo=UTC), True),
        ("4h", datetime(2026, 1, 1, 9, 0, tzinfo=UTC), False),
        ("1d", datetime(2026, 1, 1, 0, 0, tzinfo=UTC), True),
        ("1d", datetime(2026, 1, 1, 12, 0, tzinfo=UTC), False),
        ("1w", datetime(2026, 1, 5, 0, 0, tzinfo=UTC), True),
        ("1w", datetime(2026, 1, 1, 0, 0, tzinfo=UTC), False),
        ("1w", datetime(2026, 1, 5, 1, 0, tzinfo=UTC), False),
        ("1M", datetime(2026, 2, 1, 0, 0, tzinfo=UTC), True),
        ("1M", datetime(2026, 2, 2, 0, 0, tzinfo=UTC), False),
        ("1M", datetime(2026, 2, 1, 5, 0, tzinfo=UTC), False),
    ],
)
def test_candle_closes_at(timeframe, moment, closes):
    assert candle_closes_at(timeframe, moment) is closes
