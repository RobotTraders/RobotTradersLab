from datetime import UTC, datetime

import pytest

from robottraderslab._core import newest_settled_candle_start


@pytest.mark.parametrize(
    ("timeframe", "moment", "settled_start"),
    [
        (
            "15m",
            datetime(2026, 4, 1, 9, 7, 30, tzinfo=UTC),
            datetime(2026, 4, 1, 8, 45, tzinfo=UTC),
        ),
        (
            "1h",
            datetime(2026, 4, 1, 9, 59, 59, tzinfo=UTC),
            datetime(2026, 4, 1, 8, 0, tzinfo=UTC),
        ),
        (
            "1h",
            datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
            datetime(2025, 12, 31, 23, 0, tzinfo=UTC),
        ),
        (
            "4h",
            datetime(2026, 4, 1, 9, 30, tzinfo=UTC),
            datetime(2026, 4, 1, 4, 0, tzinfo=UTC),
        ),
        (
            "1d",
            datetime(2026, 4, 1, 0, 0, 1, tzinfo=UTC),
            datetime(2026, 3, 31, tzinfo=UTC),
        ),
        (
            "1w",
            datetime(2026, 4, 1, 12, 0, tzinfo=UTC),
            datetime(2026, 3, 23, tzinfo=UTC),
        ),
        (
            "1w",
            datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            datetime(2025, 12, 22, tzinfo=UTC),
        ),
        (
            "1M",
            datetime(2026, 4, 1, 0, 0, tzinfo=UTC),
            datetime(2026, 3, 1, tzinfo=UTC),
        ),
        (
            "1M",
            datetime(2026, 1, 17, 5, 0, tzinfo=UTC),
            datetime(2025, 12, 1, tzinfo=UTC),
        ),
        (
            "1d",
            datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            datetime(2025, 12, 31, tzinfo=UTC),
        ),
    ],
)
def test_the_newest_candle_whose_period_has_elapsed(timeframe, moment, settled_start):
    assert newest_settled_candle_start(timeframe, moment) == settled_start
