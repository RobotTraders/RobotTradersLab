from enum import StrEnum


class LogLevel(StrEnum):
    """Valid logging levels for the trading system."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
