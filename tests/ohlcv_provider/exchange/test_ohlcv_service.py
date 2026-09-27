import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytz

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from robottraderslab import Symbol, TimeFrame
from robottraderslab._core import DataError, OhlcvValidationError
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeTransientError,
    StrategyCriticalError,
)
from robottraderslab.exchanges import no_candles
from robottraderslab.ohlcv_provider.exceptions import PartialDownloadError
from robottraderslab.ohlcv_provider.exchange.download_metadata import (
    load_earliest_available,
    load_empty_ranges,
    store_empty_ranges,
)
from robottraderslab.ohlcv_provider.exchange.ohlcv_service import (
    fetch_ohlcv_with,
)
from robottraderslab.ohlcv_provider.repositories.null_ohlcv_repository import (
    NullOhlcvRepository,
)
from robottraderslab.ohlcv_provider.repositories.parquet_ohlcv_repository import (
    ParquetOhlcvRepository,
)

SYMBOL_BTC = Symbol.create("BTC/USDT:USDT")


HOUR_MS = 3_600_000
WEEK_MS = 7 * 24 * HOUR_MS
SATURDAY = 5


def milliseconds(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def hourly_candles(start: datetime, end: datetime) -> np.ndarray:
    stamps = np.arange(milliseconds(start), milliseconds(end) + 1, HOUR_MS)
    rows = np.full((len(stamps), 6), 1.0)
    rows[:, 0] = stamps
    return rows


def weekly_candles(first_monday: datetime, last_monday: datetime) -> np.ndarray:
    stamps = np.arange(
        milliseconds(first_monday), milliseconds(last_monday) + 1, WEEK_MS
    )
    rows = np.full((len(stamps), 6), 1.0)
    rows[:, 0] = stamps
    return rows


def weekday_candles(start: datetime, end: datetime) -> np.ndarray:
    candles = hourly_candles(start, end)
    stamps = pd.to_datetime(candles[:, 0], unit="ms", utc=True)
    return candles[stamps.dayofweek < SATURDAY]


def hourly_frame(start: datetime, end: datetime) -> pd.DataFrame:
    rows = hourly_candles(start, end)
    frame = pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    frame.index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame.drop(columns=["timestamp"])


def empty_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame.index = pd.DatetimeIndex([], tz="UTC")
    return frame


class FakeVenue:
    """Exchange source whose candles the test scripts, recording every request.

    The range named silent comes back with no rows, the shape a venue hiccup
    takes on the wire, whichever order the ranges are requested in.
    """

    def __init__(self, candles: np.ndarray, silent_from: datetime | None = None):
        self._candles = candles
        self._silent_from = silent_from
        self.requests: list[tuple[int, int]] = []

    async def __aenter__(self) -> "FakeVenue":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def fetch_ohlcv(
        self, symbol: Symbol, timeframe: TimeFrame, start_ms: int, end_ms: int
    ) -> np.ndarray:
        self.requests.append((start_ms, end_ms))
        silent = self._silent_from
        if silent is not None and start_ms == milliseconds(silent):
            return np.empty((0, 6))
        stamps = self._candles[:, 0]
        return self._candles[(stamps >= start_ms) & (stamps <= end_ms)]

    @classmethod
    def market_open_mask(cls, index: pd.DatetimeIndex) -> np.ndarray | None:
        return None


class FakeWeekendClosedVenue(FakeVenue):
    """Scripted venue whose market shuts for the weekend, as a forex feed's does."""

    @classmethod
    def market_open_mask(cls, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek < SATURDAY


class FakePartiallyFailingVenue(FakeVenue):
    """Scripted venue raising on the one range named, answering every other."""

    def __init__(self, candles: np.ndarray, failing_from: datetime):
        super().__init__(candles)
        self._failing_from = failing_from

    async def fetch_ohlcv(
        self, symbol: Symbol, timeframe: TimeFrame, start_ms: int, end_ms: int
    ) -> np.ndarray:
        if start_ms == milliseconds(self._failing_from):
            raise ExchangeTransientError("the venue dropped the connection")
        return await super().fetch_ohlcv(symbol, timeframe, start_ms, end_ms)


class FakeVenueFactory:
    """Adapter factory handing out one scripted venue."""

    def __init__(self, venue: FakeVenue):
        self._venue = venue

    @asynccontextmanager
    async def using(self, _adapter: str, **_kwargs: object) -> AsyncIterator[FakeVenue]:
        yield self._venue


class InMemoryRepository:
    """Repository keeping one series in memory."""

    def __init__(self, data: pd.DataFrame):
        self._data = data

    async def load(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        loaded = self._data
        if start_date is not None:
            loaded = loaded[loaded.index >= start_date]
        if end_date is not None:
            loaded = loaded[loaded.index <= end_date]
        return loaded.copy()

    async def store(
        self, symbol: Symbol, timeframe: TimeFrame, data: pd.DataFrame
    ) -> None:
        combined = pd.concat([self._data, data])
        self._data = combined[~combined.index.duplicated(keep="first")].sort_index()


class FakeUnreadableRepository:
    """Repository whose stored series cannot be read back."""

    async def load(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        raise OSError("the stored series is unreadable")

    async def store(
        self, symbol: Symbol, timeframe: TimeFrame, data: pd.DataFrame
    ) -> None:
        return None


async def fetch_series(
    venue: FakeVenue,
    repository: InMemoryRepository,
    start_date: datetime,
    end_date: datetime,
    storage_dir: Path,
    timeframe: TimeFrame = "1h",
) -> pd.DataFrame:
    return await fetch_ohlcv_with(
        adapter_factory=FakeVenueFactory(venue),
        repository=repository,
        adapter_name="test_exchange",
        dataset_name="test_exchange",
        symbol=SYMBOL_BTC,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        storage_dir=storage_dir,
        market_open_mask=venue.market_open_mask,
    )


def recorded_boundary(
    storage_dir: Path, timeframe: TimeFrame = "1h"
) -> datetime | None:
    return load_earliest_available(storage_dir, "test_exchange", SYMBOL_BTC, timeframe)


class TestFetchingAWindow:
    """Where a window's candles come from, and what reaches the venue."""

    @pytest.mark.parametrize(
        ("start_date", "end_date", "refusal"),
        [
            (
                datetime(2024, 1, 2, tzinfo=pytz.UTC),
                datetime(2024, 1, 1, tzinfo=pytz.UTC),
                "end_date must be after start_date",
            ),
            (None, datetime(2024, 1, 2, tzinfo=pytz.UTC), "start_date is required"),
            (datetime(2024, 1, 1, tzinfo=pytz.UTC), None, "end_date is required"),
            (
                "2024-01-01",
                datetime(2024, 1, 2, tzinfo=pytz.UTC),
                "start_date must be a datetime object",
            ),
            (
                datetime(2024, 1, 1, tzinfo=pytz.UTC),
                "2024-01-02",
                "end_date must be a datetime object",
            ),
        ],
        ids=[
            "out_of_order",
            "no_start",
            "no_end",
            "start_is_not_a_datetime",
            "end_is_not_a_datetime",
        ],
    )
    async def test_a_window_whose_dates_cannot_be_used(
        self, start_date, end_date, refusal
    ):
        with pytest.raises(OhlcvValidationError, match=refusal):
            await fetch_ohlcv_with(
                adapter_factory=FakeVenueFactory(FakeVenue(no_candles())),
                repository=InMemoryRepository(empty_frame()),
                adapter_name="test_exchange",
                dataset_name="test_exchange",
                symbol=SYMBOL_BTC,
                timeframe="1h",
                start_date=start_date,
                end_date=end_date,
                market_open_mask=FakeVenue.market_open_mask,
            )

    async def test_a_window_the_storage_covers_is_served_without_a_request(
        self, tmp_path
    ):
        january_1 = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        january_2 = datetime(2024, 1, 2, tzinfo=pytz.UTC)
        stored = hourly_frame(january_1, january_2)
        venue = FakeVenue(hourly_candles(january_1, january_2))

        served = await fetch_series(
            venue, InMemoryRepository(stored), january_1, january_2, tmp_path
        )

        assert venue.requests == []
        pd.testing.assert_frame_equal(served, stored, check_freq=False)

    async def test_a_window_the_storage_half_covers_is_completed_from_the_venue(
        self, tmp_path
    ):
        january_1 = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        january_2 = datetime(2024, 1, 2, tzinfo=pytz.UTC)
        january_3 = datetime(2024, 1, 3, tzinfo=pytz.UTC)
        repository = InMemoryRepository(hourly_frame(january_1, january_2))
        venue = FakeVenue(hourly_candles(january_1, january_3))

        served = await fetch_series(venue, repository, january_1, january_3, tmp_path)

        pd.testing.assert_frame_equal(
            served, hourly_frame(january_1, january_3), check_freq=False
        )
        pd.testing.assert_frame_equal(
            await repository.load(SYMBOL_BTC, "1h"),
            hourly_frame(january_1, january_3),
            check_freq=False,
        )

    async def test_the_candles_that_arrived_are_kept_when_a_later_range_fails(
        self, tmp_path
    ):
        january_1 = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        january_2 = datetime(2024, 1, 2, tzinfo=pytz.UTC)
        january_3 = datetime(2024, 1, 3, tzinfo=pytz.UTC)
        january_5 = datetime(2024, 1, 5, tzinfo=pytz.UTC)
        january_6 = datetime(2024, 1, 6, tzinfo=pytz.UTC)
        repository = InMemoryRepository(
            pd.concat(
                [
                    hourly_frame(january_2, january_3),
                    hourly_frame(january_5, january_6),
                ]
            )
        )
        venue = FakePartiallyFailingVenue(
            hourly_candles(january_1, january_6),
            failing_from=january_3 + timedelta(hours=1),
        )

        with pytest.raises(PartialDownloadError):
            await fetch_series(venue, repository, january_1, january_6, tmp_path)

        assert (await repository.load(SYMBOL_BTC, "1h")).index.min() == january_1

    async def test_a_storage_that_cannot_be_read(self, tmp_path):
        january_1 = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        january_2 = datetime(2024, 1, 2, tzinfo=pytz.UTC)

        with pytest.raises(DataError, match="Failed to retrieve OHLCV data"):
            await fetch_ohlcv_with(
                adapter_factory=FakeVenueFactory(FakeVenue(no_candles())),
                repository=FakeUnreadableRepository(),
                adapter_name="test_exchange",
                dataset_name="test_exchange",
                symbol=SYMBOL_BTC,
                timeframe="1h",
                start_date=january_1,
                end_date=january_2,
                storage_dir=tmp_path,
                market_open_mask=FakeVenue.market_open_mask,
            )


class TestEarliestAvailableRecording:
    """The exchange's history boundary comes only from its own answers."""

    async def test_zero_row_head_answer_leaves_the_boundary_unknown(self, tmp_path):
        may_1 = datetime(2024, 5, 1, tzinfo=pytz.UTC)
        june_1 = datetime(2024, 6, 1, tzinfo=pytz.UTC)
        june_10 = datetime(2024, 6, 10, tzinfo=pytz.UTC)
        repository = InMemoryRepository(hourly_frame(june_1, june_10))
        hiccuping_venue = FakeVenue(
            hourly_candles(datetime(2024, 4, 1, tzinfo=pytz.UTC), june_10),
            silent_from=may_1,
        )

        await fetch_series(hiccuping_venue, repository, may_1, june_10, tmp_path)

        assert recorded_boundary(tmp_path) is None

    async def test_answer_from_the_next_range_is_not_the_head_ranges_boundary(
        self, tmp_path
    ):
        may_1 = datetime(2024, 5, 1, tzinfo=pytz.UTC)
        june_1 = datetime(2024, 6, 1, tzinfo=pytz.UTC)
        june_10 = datetime(2024, 6, 10, tzinfo=pytz.UTC)
        repository = InMemoryRepository(hourly_frame(june_1, june_1))
        hiccuping_venue = FakeVenue(
            hourly_candles(datetime(2024, 4, 1, tzinfo=pytz.UTC), june_10),
            silent_from=may_1,
        )

        await fetch_series(hiccuping_venue, repository, may_1, june_10, tmp_path)

        assert sorted(start for start, _ in hiccuping_venue.requests) == [
            milliseconds(may_1),
            milliseconds(june_1),
        ]
        assert recorded_boundary(tmp_path) is None

    async def test_series_recovers_once_the_venue_answers_again(self, tmp_path):
        april_1 = datetime(2024, 4, 1, tzinfo=pytz.UTC)
        may_1 = datetime(2024, 5, 1, tzinfo=pytz.UTC)
        june_1 = datetime(2024, 6, 1, tzinfo=pytz.UTC)
        june_10 = datetime(2024, 6, 10, tzinfo=pytz.UTC)
        repository = InMemoryRepository(hourly_frame(june_1, june_10))
        hiccuping_venue = FakeVenue(hourly_candles(april_1, june_10), silent_from=may_1)
        await fetch_series(hiccuping_venue, repository, may_1, june_10, tmp_path)

        recovered = await fetch_series(
            FakeVenue(hourly_candles(april_1, june_10)),
            repository,
            april_1,
            june_10,
            tmp_path,
        )

        assert recovered.index.min() == april_1
        assert len(recovered) == len(hourly_frame(april_1, june_10))

    async def test_window_already_covered_records_no_boundary(self, tmp_path):
        june_1 = datetime(2024, 6, 1, tzinfo=pytz.UTC)
        june_10 = datetime(2024, 6, 10, tzinfo=pytz.UTC)
        repository = InMemoryRepository(hourly_frame(june_1, june_10))
        venue = FakeVenue(
            hourly_candles(datetime(2024, 4, 1, tzinfo=pytz.UTC), june_10)
        )

        await fetch_series(
            venue,
            repository,
            datetime(2024, 5, 31, 23, 30, tzinfo=pytz.UTC),
            june_10,
            tmp_path,
        )

        assert venue.requests == []
        assert recorded_boundary(tmp_path) is None

    async def test_boundary_recorded_where_the_venues_history_begins(self, tmp_path):
        april_1 = datetime(2024, 4, 1, tzinfo=pytz.UTC)
        june_10 = datetime(2024, 6, 10, tzinfo=pytz.UTC)
        repository = InMemoryRepository(empty_frame())
        venue = FakeVenue(hourly_candles(april_1, june_10))

        fetched = await fetch_series(
            venue,
            repository,
            datetime(2024, 3, 1, tzinfo=pytz.UTC),
            june_10,
            tmp_path,
        )

        assert recorded_boundary(tmp_path) == april_1
        assert fetched.index.min() == april_1

    async def test_recorded_boundary_stops_probes_before_it(self, tmp_path):
        april_1 = datetime(2024, 4, 1, tzinfo=pytz.UTC)
        june_10 = datetime(2024, 6, 10, tzinfo=pytz.UTC)
        repository = InMemoryRepository(empty_frame())
        venue = FakeVenue(hourly_candles(april_1, june_10))
        await fetch_series(
            venue,
            repository,
            datetime(2024, 3, 1, tzinfo=pytz.UTC),
            june_10,
            tmp_path,
        )
        requests_after_first_fetch = len(venue.requests)

        second_fetch = await fetch_series(
            venue,
            repository,
            datetime(2024, 2, 1, tzinfo=pytz.UTC),
            june_10,
            tmp_path,
        )

        assert len(venue.requests) == requests_after_first_fetch
        assert second_fetch.index.min() == april_1

    async def test_a_head_in_a_closed_market_is_not_a_boundary(self, tmp_path):
        april_1 = datetime(2024, 4, 1, tzinfo=pytz.UTC)
        may_1 = datetime(2024, 5, 1, tzinfo=pytz.UTC)
        saturday_afternoon = datetime(2024, 6, 1, 14, tzinfo=pytz.UTC)
        june_10 = datetime(2024, 6, 10, tzinfo=pytz.UTC)
        repository = InMemoryRepository(empty_frame())
        venue = FakeWeekendClosedVenue(weekday_candles(april_1, june_10))
        await fetch_series(venue, repository, saturday_afternoon, june_10, tmp_path)

        second_fetch = await fetch_series(venue, repository, may_1, june_10, tmp_path)

        assert milliseconds(may_1) in [start for start, _ in venue.requests]
        assert second_fetch.index.min() == may_1

    async def test_a_head_inside_trading_hours_is_still_a_boundary(self, tmp_path):
        monday = datetime(2024, 6, 3, tzinfo=pytz.UTC)
        june_10 = datetime(2024, 6, 10, tzinfo=pytz.UTC)
        repository = InMemoryRepository(empty_frame())
        venue = FakeWeekendClosedVenue(weekday_candles(monday, june_10))

        await fetch_series(
            venue,
            repository,
            datetime(2024, 5, 29, tzinfo=pytz.UTC),
            june_10,
            tmp_path,
        )

        assert recorded_boundary(tmp_path) == monday

    async def test_a_head_off_the_candle_grid_is_not_a_boundary(self, tmp_path):
        monday_january_1 = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        wednesday_march_6 = datetime(2024, 3, 6, tzinfo=pytz.UTC)
        monday_march_11 = datetime(2024, 3, 11, tzinfo=pytz.UTC)
        wednesday_june_5 = datetime(2024, 6, 5, tzinfo=pytz.UTC)
        monday_september_2 = datetime(2024, 9, 2, tzinfo=pytz.UTC)
        repository = InMemoryRepository(empty_frame())
        venue = FakeVenue(weekly_candles(monday_january_1, monday_september_2))
        await fetch_series(
            venue,
            repository,
            wednesday_june_5,
            monday_september_2,
            tmp_path,
            timeframe="1w",
        )

        second_fetch = await fetch_series(
            venue,
            repository,
            wednesday_march_6,
            monday_september_2,
            tmp_path,
            timeframe="1w",
        )

        assert milliseconds(wednesday_march_6) in [start for start, _ in venue.requests]
        assert second_fetch.index.min() == monday_march_11

    async def test_a_boundary_a_whole_candle_after_an_off_grid_head_is_recorded(
        self, tmp_path
    ):
        wednesday_june_5 = datetime(2024, 6, 5, tzinfo=pytz.UTC)
        monday_june_17 = datetime(2024, 6, 17, tzinfo=pytz.UTC)
        monday_september_2 = datetime(2024, 9, 2, tzinfo=pytz.UTC)
        repository = InMemoryRepository(empty_frame())
        venue = FakeVenue(weekly_candles(monday_june_17, monday_september_2))

        await fetch_series(
            venue,
            repository,
            wednesday_june_5,
            monday_september_2,
            tmp_path,
            timeframe="1w",
        )

        assert recorded_boundary(tmp_path, "1w") == monday_june_17

    async def test_a_boundary_one_candle_after_the_head_is_recorded(self, tmp_path):
        june_1 = datetime(2024, 6, 1, tzinfo=pytz.UTC)
        june_1_at_one = june_1 + timedelta(hours=1)
        june_10 = datetime(2024, 6, 10, tzinfo=pytz.UTC)
        repository = InMemoryRepository(empty_frame())
        venue = FakeVenue(hourly_candles(june_1_at_one, june_10))

        await fetch_series(venue, repository, june_1, june_10, tmp_path)

        assert recorded_boundary(tmp_path) == june_1_at_one

    async def test_gap_inside_the_history_is_not_a_boundary(self, tmp_path):
        january_1 = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        march_10 = datetime(2024, 3, 10, tzinfo=pytz.UTC)
        cached = pd.concat(
            [
                hourly_frame(january_1, datetime(2024, 1, 31, tzinfo=pytz.UTC)),
                hourly_frame(datetime(2024, 3, 1, tzinfo=pytz.UTC), march_10),
            ]
        )
        repository = InMemoryRepository(cached)
        venue = FakeVenue(
            hourly_candles(datetime(2024, 2, 15, tzinfo=pytz.UTC), march_10)
        )

        await fetch_series(venue, repository, january_1, march_10, tmp_path)

        assert recorded_boundary(tmp_path) is None


class TestEmptyRangeWatermark:
    """Closed periods the exchange reports as empty are not requested again."""

    async def test_closed_empty_range_is_recorded(self, tmp_path):
        january_1 = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        january_2 = datetime(2024, 1, 2, tzinfo=pytz.UTC)
        silent_venue = FakeVenue(no_candles())

        await fetch_series(
            silent_venue,
            InMemoryRepository(empty_frame()),
            january_1,
            january_2,
            tmp_path,
        )

        assert load_empty_ranges(tmp_path, "test_exchange", SYMBOL_BTC, "1h") == [
            (january_1, january_2)
        ]

    async def test_recorded_range_is_not_requested_again(self, tmp_path):
        january_1 = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        january_2 = datetime(2024, 1, 2, tzinfo=pytz.UTC)
        store_empty_ranges(
            tmp_path, "test_exchange", SYMBOL_BTC, "1h", [(january_1, january_2)]
        )
        venue = FakeVenue(hourly_candles(january_1, january_2))

        await fetch_series(
            venue, InMemoryRepository(empty_frame()), january_1, january_2, tmp_path
        )

        assert venue.requests == []

    async def test_range_reaching_the_forming_candle_is_not_recorded(self, tmp_path):
        now = datetime.now(timezone.utc)
        silent_venue = FakeVenue(no_candles())

        await fetch_series(
            silent_venue,
            InMemoryRepository(empty_frame()),
            now - timedelta(hours=3),
            now,
            tmp_path,
        )

        assert load_empty_ranges(tmp_path, "test_exchange", SYMBOL_BTC, "1h") == []


class _FailingVenue:
    """Exchange source whose every request raises the failure it was given."""

    def __init__(self, failure: Exception):
        self._failure = failure

    async def __aenter__(self) -> "_FailingVenue":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def fetch_ohlcv(
        self, symbol: Symbol, timeframe: TimeFrame, start_ms: int, end_ms: int
    ) -> np.ndarray:
        raise self._failure

    @classmethod
    def market_open_mask(cls, index: pd.DatetimeIndex) -> np.ndarray | None:
        return None


class TestCriticalsFromTheVenue:
    @pytest.mark.parametrize(
        "failure",
        [
            ExchangeCriticalError("key revoked"),
            StrategyCriticalError("SOL/USD:USD is not a Kraken perpetual"),
        ],
        ids=["exchange_critical", "strategy_critical"],
    )
    async def test_a_critical_the_venue_raises_reaches_the_caller_untouched(
        self, failure
    ):
        with pytest.raises(type(failure), match=str(failure)):
            await fetch_ohlcv_with(
                adapter_factory=FakeVenueFactory(_FailingVenue(failure)),
                repository=InMemoryRepository(empty_frame()),
                adapter_name="test_exchange",
                dataset_name="test_exchange",
                symbol=SYMBOL_BTC,
                timeframe="1h",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
                market_open_mask=_FailingVenue.market_open_mask,
            )


class TestServingWithoutStorage:
    """A storage keeping nothing still serves the candles that arrived."""

    async def test_the_downloaded_window_is_served_when_nothing_is_kept(self):
        june_1 = datetime(2024, 6, 1, tzinfo=pytz.UTC)
        june_3 = datetime(2024, 6, 3, tzinfo=pytz.UTC)
        venue = FakeVenue(hourly_candles(june_1, june_3))

        served = await fetch_ohlcv_with(
            adapter_factory=FakeVenueFactory(venue),
            repository=NullOhlcvRepository(),
            adapter_name="test_exchange",
            dataset_name="test_exchange",
            symbol=SYMBOL_BTC,
            timeframe="1h",
            start_date=june_1,
            end_date=june_3,
            market_open_mask=venue.market_open_mask,
        )

        pd.testing.assert_frame_equal(
            served, hourly_frame(june_1, june_3), check_freq=False
        )


def hourly_frame_with_holes(
    start: datetime, end: datetime, holes: list[datetime]
) -> pd.DataFrame:
    frame = hourly_frame(start, end)
    frame.loc[holes] = np.nan
    return frame


def parquet_repository(storage_dir: Path) -> ParquetOhlcvRepository:
    return ParquetOhlcvRepository("test_exchange", storage_dir)


class TestStoredHoles:
    """A stored row no candle stands behind is asked of the venue again."""

    async def test_a_hole_inside_market_hours_is_requested(self, tmp_path):
        january_1 = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        january_2 = datetime(2024, 1, 2, tzinfo=pytz.UTC)
        hole = january_1 + timedelta(hours=6)
        repository = parquet_repository(tmp_path)
        await repository.store(
            SYMBOL_BTC, "1h", hourly_frame_with_holes(january_1, january_2, [hole])
        )
        venue = FakeVenue(hourly_candles(january_1, january_2))

        await fetch_series(venue, repository, january_1, january_2, tmp_path)

        assert venue.requests == [
            (milliseconds(hole), milliseconds(hole + timedelta(hours=1)))
        ]

    async def test_the_venues_candle_replaces_the_hole(self, tmp_path):
        january_1 = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        january_2 = datetime(2024, 1, 2, tzinfo=pytz.UTC)
        hole = january_1 + timedelta(hours=6)
        repository = parquet_repository(tmp_path)
        await repository.store(
            SYMBOL_BTC, "1h", hourly_frame_with_holes(january_1, january_2, [hole])
        )
        venue = FakeVenue(hourly_candles(january_1, january_2))
        await fetch_series(venue, repository, january_1, january_2, tmp_path)

        healed = await fetch_series(venue, repository, january_1, january_2, tmp_path)

        assert len(venue.requests) == 1
        assert healed.loc[hole, "close"] == 1.0

    async def test_a_hole_the_venue_answers_without_is_asked_once(self, tmp_path):
        january_1 = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        january_2 = datetime(2024, 1, 2, tzinfo=pytz.UTC)
        hole = january_1 + timedelta(hours=6)
        repository = parquet_repository(tmp_path)
        await repository.store(
            SYMBOL_BTC, "1h", hourly_frame_with_holes(january_1, january_2, [hole])
        )
        candles = hourly_candles(january_1, january_2)
        venue = FakeVenue(candles[candles[:, 0] != milliseconds(hole)])
        await fetch_series(venue, repository, january_1, january_2, tmp_path)

        served_again = await fetch_series(
            venue, repository, january_1, january_2, tmp_path
        )

        assert load_empty_ranges(tmp_path, "test_exchange", SYMBOL_BTC, "1h") == [
            (hole, hole + timedelta(hours=1))
        ]
        assert len(venue.requests) == 1
        assert np.isnan(served_again.loc[hole, "close"])

    async def test_a_hole_on_the_windows_end_is_asked_once(self, tmp_path):
        january_1 = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        january_2 = datetime(2024, 1, 2, tzinfo=pytz.UTC)
        noon = january_1 + timedelta(hours=12)
        repository = parquet_repository(tmp_path)
        await repository.store(
            SYMBOL_BTC, "1h", hourly_frame_with_holes(january_1, january_2, [noon])
        )
        candles = hourly_candles(january_1, january_2)
        venue = FakeVenue(candles[candles[:, 0] != milliseconds(noon)])
        await fetch_series(venue, repository, january_1, noon, tmp_path)

        await fetch_series(venue, repository, january_1, noon, tmp_path)

        assert load_empty_ranges(tmp_path, "test_exchange", SYMBOL_BTC, "1h") == [
            (noon - timedelta(hours=1), noon)
        ]
        assert len(venue.requests) == 1

    async def test_a_hole_outside_market_hours_is_not_requested(self, tmp_path):
        friday = datetime(2024, 1, 5, tzinfo=pytz.UTC)
        monday = datetime(2024, 1, 8, tzinfo=pytz.UTC)
        weekend = [
            stamp.to_pydatetime()
            for stamp in pd.date_range(friday, monday, freq="1h")
            if stamp.dayofweek >= SATURDAY
        ]
        repository = parquet_repository(tmp_path)
        await repository.store(
            SYMBOL_BTC, "1h", hourly_frame_with_holes(friday, monday, weekend)
        )
        venue = FakeWeekendClosedVenue(weekday_candles(friday, monday))

        await fetch_series(venue, repository, friday, monday, tmp_path)

        assert venue.requests == []
