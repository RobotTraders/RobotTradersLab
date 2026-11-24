from pathlib import Path

import pandas as pd

from ..date_utils import standardize_ohlcv_index
from .file_based_ohlcv_repository import FileBasedOhlcvRepository


class ParquetOhlcvRepository(FileBasedOhlcvRepository):
    """Parquet-based implementation of OHLCV data repository."""

    @property
    def _file_extension(self) -> str:
        return ".parquet"

    def _read_file(self, file_path: Path) -> pd.DataFrame:
        data = pd.read_parquet(file_path)
        return standardize_ohlcv_index(data)

    def _write_file(self, data: pd.DataFrame, file_path: Path) -> None:
        data.to_parquet(file_path, index=True)
