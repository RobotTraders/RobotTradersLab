from .timeframes import TimeFrame


class ExchangeRecoverableError(Exception):
    """The operation failed but the system can continue with other operations."""

    pass


class ExchangeTransientError(ExchangeRecoverableError):
    """A recoverable failure worth retrying before it is treated as one.

    Covers 5xx responses, connection drops, and timeouts: conditions that
    are likely to resolve on a subsequent attempt.
    """

    pass


class MissingOhlcvDataError(ExchangeRecoverableError):
    """A symbol/timeframe pair was read that this run's candles do not hold."""

    def __init__(
        self,
        symbol: object,
        timeframe: object,
        detail: str = "no candles were loaded",
    ) -> None:
        """
        Args:
            detail: What the run's candles lack for the pair.
        """
        super().__init__(f"{symbol}@{timeframe}: {detail}")


class NoOpenPositionError(ExchangeRecoverableError):
    """A protection update named a symbol the account holds no position on,
    so there is nothing left to protect.
    """

    pass


class ExchangeCriticalError(BaseException):
    """The venue is unusable, so trading stops until someone intervenes.

    It stands outside the `Exception` tree so that no `except Exception`
    between the venue and the entry point can keep the run going on it.
    """

    pass


class StrategyCriticalError(BaseException):
    """The strategy's configuration or code is broken, so trading stops.

    It stands outside the `Exception` tree so that no `except Exception`
    between the strategy and the entry point can keep the run going on it.
    """

    pass


class DataError(Exception):
    """The candles could not be supplied: a source that is unreadable or
    empty, one holding no candle in the window asked for, a request the
    provider refuses, or a download that failed.
    """

    pass


class OhlcvValidationError(DataError):
    """A candle request the provider refuses as asked."""

    pass


class DownloadError(DataError, ExchangeTransientError):
    """A candle download failed; transient, so the fetch is retried."""

    pass


class ExchangeConnectionError(DownloadError):
    """The venue could not be reached for the pair's candles; transient, so
    the fetch is retried.
    """

    def __init__(self, symbol: object, timeframe: object, detail: str) -> None:
        super().__init__(
            f"Exchange connection error for {symbol}/{timeframe}: {detail}"
        )


class RateLimitExceededError(DownloadError):
    """The venue refused a candle request for exceeding its rate limit;
    transient, so the fetch is retried.
    """

    def __init__(self, symbol: object, timeframe: object, detail: str) -> None:
        super().__init__(f"Rate limit exceeded for {symbol}/{timeframe}: {detail}")


class InvalidTimeframeError(DataError, ExchangeRecoverableError):
    """The venue serves no candles on the timeframe; the message lists the
    ones it serves.
    """

    def __init__(self, timeframe: object, supported: set[TimeFrame]) -> None:
        supported_str = ", ".join(sorted(map(str, supported)))
        super().__init__(
            f"Timeframe '{timeframe}' not supported by exchange. Supported timeframes: {supported_str}"
        )
