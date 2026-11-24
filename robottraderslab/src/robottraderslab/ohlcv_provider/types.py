from datetime import datetime
from enum import StrEnum

type DateRange = tuple[datetime, datetime]
type DateRanges = list[DateRange]


class OhlcvColumn(StrEnum):
    TIMESTAMP = "timestamp"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
