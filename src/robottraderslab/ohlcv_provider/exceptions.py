import pandas as pd

from robottraderslab._core import (
    DataError,
    DownloadError,
    ExchangeRecoverableError,
)


class PartialDownloadError(DownloadError):
    """Raised when part of a download succeeded before another part failed.

    Carries what did arrive so the caller can persist it and resume the
    download from there.
    """

    def __init__(self, message: str, data: pd.DataFrame) -> None:
        self.data = data
        super().__init__(message)


class InvalidSymbolError(DataError, ExchangeRecoverableError):
    """Symbol not supported by exchange."""

    def __init__(self, symbol: object) -> None:
        super().__init__(f"Symbol '{symbol}' not supported by exchange")


class StorageTypeError(DataError):
    """Unsupported storage type."""

    def __init__(self, storage_type: object) -> None:
        super().__init__(f"Unsupported storage_type: {storage_type}")
