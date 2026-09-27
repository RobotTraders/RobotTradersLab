from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [101.0, 102.0, 103.0, 104.0, 105.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        },
        index=dates,
    )


@pytest.fixture
def additional_ohlcv_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01 05:00:00", periods=3, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [105.0, 106.0, 107.0],
            "high": [110.0, 111.0, 112.0],
            "low": [104.0, 105.0, 106.0],
            "close": [106.0, 107.0, 108.0],
            "volume": [1500.0, 1600.0, 1700.0],
        },
        index=dates,
    )


@pytest.fixture
def temp_storage_dir() -> Iterator[Path]:
    with TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
