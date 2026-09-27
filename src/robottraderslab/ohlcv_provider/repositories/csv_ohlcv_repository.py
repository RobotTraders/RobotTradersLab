from pathlib import Path

import pandas as pd

from ..date_utils import get_date_format, is_timezone_aware, standardize_ohlcv_index
from .file_based_ohlcv_repository import FileBasedOhlcvRepository


class CsvOhlcvRepository(FileBasedOhlcvRepository):
    """CSV-based implementation of OHLCV data repository."""

    @property
    def _file_extension(self) -> str:
        return ".csv"

    def _read_file(self, file_path: Path) -> pd.DataFrame:
        data = pd.read_csv(
            file_path,
            engine="c",
            index_col="date",
            parse_dates=True,
            date_format="ISO8601" if is_timezone_aware() else None,
            dtype={
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
            },
            na_filter=False,
        )

        return standardize_ohlcv_index(data)

    def _write_file(self, data: pd.DataFrame, file_path: Path) -> None:
        data.to_csv(
            file_path,
            mode="w",
            header=True,
            index=True,
            index_label="date",
            date_format=get_date_format(),
        )
