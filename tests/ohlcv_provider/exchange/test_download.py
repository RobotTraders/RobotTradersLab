import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytz

from robottraderslab import Symbol, TimeFrame
from robottraderslab._core import DataError, DownloadError
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeTransientError,
    StrategyCriticalError,
)
from robottraderslab.exchanges import OhlcvAdapterProtocol, OhlcvData, no_candles
from robottraderslab.ohlcv_provider.exceptions import PartialDownloadError
from robottraderslab.ohlcv_provider.exchange.download import fetch_missing_data

JANUARY_FIRST_MS = 1704067200000
HOUR_MS = 3_600_000

ONE_RANGE = [
    (
        datetime(2024, 1, 1, tzinfo=pytz.UTC),
        datetime(2024, 1, 1, 5, tzinfo=pytz.UTC),
    )
]

TWO_RANGES = [
    (
        datetime(2024, 1, 1, tzinfo=pytz.UTC),
        datetime(2024, 1, 1, 5, tzinfo=pytz.UTC),
    ),
    (
        datetime(2024, 1, 2, tzinfo=pytz.UTC),
        datetime(2024, 1, 2, 5, tzinfo=pytz.UTC),
    ),
]


@pytest.fixture
def mock_adapter() -> AsyncMock:
    adapter = AsyncMock(spec=OhlcvAdapterProtocol)
    adapter.fetch_ohlcv.return_value = no_candles()
    adapter.market_open_mask.return_value = None
    return adapter


@pytest.fixture
def sample_symbol() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def sample_timeframe() -> TimeFrame:
    return "1h"


@pytest.fixture
def sample_ohlcv_rows() -> OhlcvData:
    return np.array(
        [
            [JANUARY_FIRST_MS, 100.0, 105.0, 95.0, 101.0, 1000.0],
            [JANUARY_FIRST_MS + HOUR_MS, 101.0, 106.0, 96.0, 102.0, 1100.0],
            [JANUARY_FIRST_MS + 2 * HOUR_MS, 102.0, 107.0, 97.0, 103.0, 1200.0],
            [JANUARY_FIRST_MS + 3 * HOUR_MS, 103.0, 108.0, 98.0, 104.0, 1300.0],
            [JANUARY_FIRST_MS + 4 * HOUR_MS, 104.0, 109.0, 99.0, 105.0, 1400.0],
        ]
    )


@pytest.fixture
def gapped_ohlcv_rows(sample_ohlcv_rows: OhlcvData) -> OhlcvData:
    missing_candle = JANUARY_FIRST_MS + 2 * HOUR_MS
    return sample_ohlcv_rows[sample_ohlcv_rows[:, 0] != missing_candle]


class TestFetchMissingDataSuccess:
    async def test_fetches_single_range_successfully(
        self,
        mock_adapter,
        sample_symbol,
        sample_timeframe,
        sample_ohlcv_rows,
    ):
        mock_adapter.fetch_ohlcv.return_value = sample_ohlcv_rows

        candles = await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            ONE_RANGE,
        )

        assert len(candles) == 5

    async def test_asks_the_adapter_for_the_whole_range(
        self,
        mock_adapter,
        sample_symbol,
        sample_timeframe,
        sample_ohlcv_rows,
    ):
        mock_adapter.fetch_ohlcv.return_value = sample_ohlcv_rows

        await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            ONE_RANGE,
        )

        requested = mock_adapter.fetch_ohlcv.call_args
        range_start, range_end = ONE_RANGE[0]
        assert requested.args[2] == int(range_start.timestamp() * 1000)
        assert requested.args[3] == int(range_end.timestamp() * 1000)

    async def test_fetches_multiple_ranges_successfully(
        self,
        mock_adapter,
        sample_symbol,
        sample_timeframe,
        sample_ohlcv_rows,
    ):
        mock_adapter.fetch_ohlcv.return_value = sample_ohlcv_rows

        candles = await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            TWO_RANGES,
        )

        assert not candles.empty
        assert mock_adapter.fetch_ohlcv.call_count == 2

    async def test_combines_and_deduplicates_data_from_multiple_ranges(
        self, mock_adapter, sample_symbol, sample_timeframe
    ):
        overlapping_rows = np.array(
            [
                [JANUARY_FIRST_MS, 100.0, 105.0, 95.0, 101.0, 1000.0],
                [JANUARY_FIRST_MS + HOUR_MS, 101.0, 106.0, 96.0, 102.0, 1100.0],
                [JANUARY_FIRST_MS + 2 * HOUR_MS, 102.0, 107.0, 97.0, 103.0, 1200.0],
            ]
        )
        mock_adapter.fetch_ohlcv.return_value = overlapping_rows

        candles = await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            [
                (
                    datetime(2024, 1, 1, tzinfo=pytz.UTC),
                    datetime(2024, 1, 1, 2, tzinfo=pytz.UTC),
                ),
                (
                    datetime(2024, 1, 1, 1, tzinfo=pytz.UTC),
                    datetime(2024, 1, 1, 3, tzinfo=pytz.UTC),
                ),
            ],
        )

        assert len(candles) == 3
        assert candles.index.is_unique

    async def test_rows_arriving_out_of_order(
        self,
        mock_adapter,
        sample_symbol,
        sample_timeframe,
        sample_ohlcv_rows,
    ):
        mock_adapter.fetch_ohlcv.return_value = sample_ohlcv_rows[::-1]

        candles = await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            ONE_RANGE,
        )

        assert candles.index.is_monotonic_increasing
        assert candles["close"].tolist() == [101.0, 102.0, 103.0, 104.0, 105.0]


class TestGapFillPersistence:
    async def test_keeps_every_gap_fill_row(
        self, mock_adapter, sample_symbol, sample_timeframe
    ):
        rows_with_two_gaps = np.array(
            [
                [JANUARY_FIRST_MS, 100.0, 105.0, 95.0, 101.0, 1000.0],
                [JANUARY_FIRST_MS + 3 * HOUR_MS, 103.0, 108.0, 98.0, 104.0, 1300.0],
                [JANUARY_FIRST_MS + 5 * HOUR_MS, 104.0, 109.0, 99.0, 105.0, 1400.0],
            ]
        )
        mock_adapter.fetch_ohlcv.return_value = rows_with_two_gaps

        candles = await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            ONE_RANGE,
        )

        assert len(candles) == 6
        assert int(candles.isna().any(axis=1).sum()) == 3

    async def test_keeps_candles_with_identical_values(
        self, mock_adapter, sample_symbol, sample_timeframe
    ):
        flat_market_rows = np.array(
            [
                [JANUARY_FIRST_MS, 100.0, 100.0, 100.0, 100.0, 0.0],
                [JANUARY_FIRST_MS + HOUR_MS, 100.0, 100.0, 100.0, 100.0, 0.0],
                [JANUARY_FIRST_MS + 2 * HOUR_MS, 100.0, 100.0, 100.0, 100.0, 0.0],
            ]
        )
        mock_adapter.fetch_ohlcv.return_value = flat_market_rows

        candles = await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            [
                (
                    datetime(2024, 1, 1, tzinfo=pytz.UTC),
                    datetime(2024, 1, 1, 2, tzinfo=pytz.UTC),
                )
            ],
        )

        assert len(candles) == 3


class TestFetchMissingDataEdgeCases:
    async def test_no_ranges_provided(
        self, mock_adapter, sample_symbol, sample_timeframe
    ):
        candles = await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            [],
        )

        assert candles.empty

    async def test_every_range_comes_back_empty(
        self, mock_adapter, sample_symbol, sample_timeframe
    ):
        candles = await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            TWO_RANGES,
        )

        assert candles.empty

    async def test_one_range_comes_back_empty(
        self,
        mock_adapter,
        sample_symbol,
        sample_timeframe,
        sample_ohlcv_rows,
    ):
        mock_adapter.fetch_ohlcv.side_effect = [sample_ohlcv_rows, no_candles()]

        candles = await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            TWO_RANGES,
        )

        assert len(candles) == 5

    async def test_same_start_and_end_date(
        self, mock_adapter, sample_symbol, sample_timeframe
    ):
        candles = await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            [
                (
                    datetime(2024, 1, 1, tzinfo=pytz.UTC),
                    datetime(2024, 1, 1, tzinfo=pytz.UTC),
                )
            ],
        )

        assert candles.empty

    async def test_end_date_before_start_date(
        self, mock_adapter, sample_symbol, sample_timeframe
    ):
        candles = await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            [
                (
                    datetime(2024, 1, 2, tzinfo=pytz.UTC),
                    datetime(2024, 1, 1, tzinfo=pytz.UTC),
                )
            ],
        )

        assert candles.empty

    async def test_unsupported_timeframe(self, mock_adapter, sample_symbol):
        with pytest.raises(Exception, match="Unsupported timeframe"):
            await fetch_missing_data(
                mock_adapter,
                sample_symbol,
                "invalid_timeframe",
                ONE_RANGE,
            )


class TestFetchMissingDataErrorHandling:
    async def test_a_range_that_fails_on_a_transient_error(
        self, mock_adapter, sample_symbol, sample_timeframe
    ):
        mock_adapter.fetch_ohlcv.side_effect = ExchangeTransientError("Network error")

        with pytest.raises(DownloadError, match="failed to fetch from exchange"):
            await fetch_missing_data(
                mock_adapter,
                sample_symbol,
                sample_timeframe,
                TWO_RANGES,
            )

    async def test_a_range_that_fails_on_an_error_a_retry_cannot_fix(
        self, mock_adapter, sample_symbol, sample_timeframe
    ):
        mock_adapter.fetch_ohlcv.side_effect = Exception("symbol not listed")

        with pytest.raises(DataError, match="failed to fetch from exchange"):
            await fetch_missing_data(
                mock_adapter,
                sample_symbol,
                sample_timeframe,
                TWO_RANGES,
            )

    @pytest.mark.parametrize(
        "failure",
        [
            ExchangeCriticalError("key revoked"),
            StrategyCriticalError("key revoked"),
        ],
        ids=["exchange_critical", "strategy_critical"],
    )
    async def test_a_critical_error_propagates_untouched(
        self, mock_adapter, sample_symbol, sample_timeframe, failure
    ):
        mock_adapter.fetch_ohlcv.side_effect = failure

        with pytest.raises(type(failure), match="key revoked"):
            await fetch_missing_data(
                mock_adapter,
                sample_symbol,
                sample_timeframe,
                ONE_RANGE,
            )

    async def test_cancellation_leaves_no_pending_download_tasks(
        self, mock_adapter, sample_symbol, sample_timeframe
    ):
        first_range_failed = asyncio.Event()
        release_second_range = asyncio.Event()

        async def fetch_ohlcv(symbol, timeframe, start_ms, end_ms):
            if not first_range_failed.is_set():
                first_range_failed.set()
                raise Exception("Network error")
            await release_second_range.wait()
            return no_candles()

        mock_adapter.fetch_ohlcv.side_effect = fetch_ohlcv

        download = asyncio.create_task(
            fetch_missing_data(
                mock_adapter,
                sample_symbol,
                sample_timeframe,
                TWO_RANGES,
            )
        )
        await first_range_failed.wait()
        download.cancel()
        with pytest.raises(asyncio.CancelledError):
            await download

        leftover = asyncio.all_tasks() - {asyncio.current_task()}
        assert not leftover

    async def test_failure_keeps_the_candles_already_fetched(
        self,
        mock_adapter,
        sample_symbol,
        sample_timeframe,
        sample_ohlcv_rows,
    ):
        first_range_done = asyncio.Event()

        async def fetch_ohlcv(symbol, timeframe, start_ms, end_ms):
            if not first_range_done.is_set():
                first_range_done.set()
                return sample_ohlcv_rows
            raise Exception("Network error")

        mock_adapter.fetch_ohlcv.side_effect = fetch_ohlcv

        with pytest.raises(PartialDownloadError) as failure:
            await fetch_missing_data(
                mock_adapter,
                sample_symbol,
                sample_timeframe,
                TWO_RANGES,
            )

        assert not failure.value.data.empty


class TestFetchMissingDataIntegration:
    async def test_preserves_data_integrity_through_pipeline(
        self, mock_adapter, sample_symbol, sample_timeframe
    ):
        rows = np.array(
            [
                [JANUARY_FIRST_MS, 100.0, 105.0, 95.0, 100.5, 1000.0],
                [JANUARY_FIRST_MS + HOUR_MS, 101.0, 106.0, 96.0, 101.5, 1100.0],
                [JANUARY_FIRST_MS + 2 * HOUR_MS, 102.0, 107.0, 97.0, 102.5, 1200.0],
            ]
        )
        mock_adapter.fetch_ohlcv.return_value = rows

        candles = await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            [
                (
                    datetime(2024, 1, 1, tzinfo=pytz.UTC),
                    datetime(2024, 1, 1, 3, tzinfo=pytz.UTC),
                )
            ],
        )

        assert len(candles) == 3
        assert candles["close"].tolist() == [100.5, 101.5, 102.5]
        assert candles["volume"].tolist() == [1000.0, 1100.0, 1200.0]
        assert str(candles.index.tz).upper() == "UTC"

    async def test_handles_large_number_of_ranges(
        self,
        mock_adapter,
        sample_symbol,
        sample_timeframe,
        sample_ohlcv_rows,
    ):
        mock_adapter.fetch_ohlcv.return_value = sample_ohlcv_rows

        await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            [
                (
                    datetime(2024, 1, day, tzinfo=pytz.UTC),
                    datetime(2024, 1, day, 5, tzinfo=pytz.UTC),
                )
                for day in range(1, 11)
            ],
        )

        assert mock_adapter.fetch_ohlcv.call_count == 10

    async def test_candles_missing_inside_a_range(
        self,
        mock_adapter,
        sample_symbol,
        sample_timeframe,
        gapped_ohlcv_rows,
    ):
        mock_adapter.fetch_ohlcv.return_value = gapped_ohlcv_rows

        candles = await fetch_missing_data(
            mock_adapter,
            sample_symbol,
            sample_timeframe,
            ONE_RANGE,
        )

        assert len(candles) == 5
