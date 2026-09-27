from datetime import datetime

import pytest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import (
    EquitySnapshot,
    SimulatedPosition,
)
from robottraderslab.backtester.simulator.daily_equity_recorder import (
    DailyEquityRecorder,
)
from robottraderslab.exchanges import Currency, PositionSide

ZERO_POINT_ONE_BTC_IN_STEPS = 10_000_000


@pytest.fixture
def daily_equity_recorder() -> DailyEquityRecorder:
    return DailyEquityRecorder()


@pytest.fixture
def balances() -> dict[Currency, float]:
    return {"USDT": 10_000.0}


@pytest.fixture
def sample_positions(btc_usdt_perp: Symbol) -> dict[Symbol, SimulatedPosition]:
    btc_position = SimulatedPosition(
        symbol=btc_usdt_perp,
        side=PositionSide.LONG,
        leverage=1.0,
        taker_fee_rate=0.001,
        entry_time=datetime.fromisoformat("2024-01-01 10:00:00"),
    )
    btc_position.add_to_position(ZERO_POINT_ONE_BTC_IN_STEPS, 50_000.0)
    return {btc_usdt_perp: btc_position}


@pytest.fixture
def futures_prices(btc_usdt_perp: Symbol, eth_usdt_perp: Symbol) -> dict[Symbol, float]:
    return {btc_usdt_perp: 55_000.0, eth_usdt_perp: 3000.0}


class TestConstruction:
    def test_construction(self):
        daily_equity_recorder = DailyEquityRecorder()

        assert daily_equity_recorder.snapshots == []


class TestEquitySnapshotRecording:
    def test_daily_at_midnight(
        self, daily_equity_recorder, balances, sample_positions, futures_prices
    ):
        timestamp = datetime.fromisoformat("2024-01-01 00:00:00")
        snapshot = EquitySnapshot(
            timestamp=timestamp,
            market_type="futures",
            balances=balances,
            positions=sample_positions,
            prices=futures_prices,
        )

        daily_equity_recorder.record_snapshot(snapshot)

        assert len(daily_equity_recorder.snapshots) == 1
        snapshot = daily_equity_recorder.snapshots[0]
        assert snapshot.timestamp == timestamp
        assert snapshot.market_type == "futures"
        assert snapshot.balances == balances
        assert snapshot.positions == sample_positions
        assert snapshot.prices == futures_prices

    def test_daily_not_at_midnight(
        self, daily_equity_recorder, balances, sample_positions, futures_prices
    ):
        timestamp = datetime.fromisoformat("2024-01-01 12:30:00")
        snapshot = EquitySnapshot(
            timestamp=timestamp,
            market_type="futures",
            balances=balances,
            positions=sample_positions,
            prices=futures_prices,
        )

        daily_equity_recorder.record_snapshot(snapshot)

        assert len(daily_equity_recorder.snapshots) == 0

    def test_multiple_midnight_snapshots(
        self, daily_equity_recorder, balances, sample_positions, futures_prices
    ):
        timestamps = [
            datetime.fromisoformat("2024-01-01 00:00:00"),
            datetime.fromisoformat("2024-01-02 00:00:00"),
            datetime.fromisoformat("2024-01-03 00:00:00"),
        ]

        for timestamp in timestamps:
            snapshot = EquitySnapshot(
                timestamp=timestamp,
                market_type="futures",
                balances=balances,
                positions=sample_positions,
                prices=futures_prices,
            )
            daily_equity_recorder.record_snapshot(snapshot)

        assert len(daily_equity_recorder.snapshots) == 3

    def test_ignores_non_midnight_timestamps(
        self, daily_equity_recorder, balances, sample_positions, futures_prices
    ):
        non_midnight_timestamps = [
            datetime.fromisoformat("2024-01-01 09:15:00"),
            datetime.fromisoformat("2024-01-01 14:30:00"),
            datetime.fromisoformat("2024-01-01 23:59:59"),
        ]

        for timestamp in non_midnight_timestamps:
            snapshot = EquitySnapshot(
                timestamp=timestamp,
                market_type="futures",
                balances=balances,
                positions=sample_positions,
                prices=futures_prices,
            )
            daily_equity_recorder.record_snapshot(snapshot)

        assert len(daily_equity_recorder.snapshots) == 0

    @pytest.mark.parametrize(
        ("hour", "minute", "second", "should_record"),
        [
            (0, 0, 0, True),
            (0, 0, 1, False),
            (0, 1, 0, False),
            (1, 0, 0, False),
            (23, 59, 59, False),
        ],
    )
    def test_midnight_precision(
        self,
        daily_equity_recorder,
        balances,
        sample_positions,
        futures_prices,
        hour,
        minute,
        second,
        should_record,
    ):
        timestamp = datetime.fromisoformat(
            f"2024-01-01 {hour:02d}:{minute:02d}:{second:02d}"
        )
        snapshot = EquitySnapshot(
            timestamp=timestamp,
            market_type="futures",
            balances=balances,
            positions=sample_positions,
            prices=futures_prices,
        )

        daily_equity_recorder.record_snapshot(snapshot)

        expected_count = 1 if should_record else 0
        assert len(daily_equity_recorder.snapshots) == expected_count


class TestRunBounds:
    def test_a_bound_off_midnight_is_kept(
        self, daily_equity_recorder, balances, sample_positions, futures_prices
    ):
        timestamp = datetime.fromisoformat("2024-01-01 00:15:00")
        snapshot = EquitySnapshot(
            timestamp=timestamp,
            market_type="futures",
            balances=balances,
            positions=sample_positions,
            prices=futures_prices,
        )

        daily_equity_recorder.record_bound(snapshot)

        assert [kept.timestamp for kept in daily_equity_recorder.snapshots] == [
            timestamp
        ]

    def test_a_bound_on_a_recorded_midnight_takes_its_place(
        self, daily_equity_recorder, sample_positions, futures_prices
    ):
        midnight = datetime.fromisoformat("2024-01-02 00:00:00")
        before_booking = EquitySnapshot(
            timestamp=midnight,
            market_type="futures",
            balances={"USDT": 10_000.0},
            positions=sample_positions,
            prices=futures_prices,
        )
        closing = EquitySnapshot(
            timestamp=midnight,
            market_type="futures",
            balances={"USDT": 9_999.0},
            positions=sample_positions,
            prices=futures_prices,
        )

        daily_equity_recorder.record_snapshot(before_booking)
        daily_equity_recorder.record_bound(closing)

        assert len(daily_equity_recorder.snapshots) == 1
        assert daily_equity_recorder.snapshots[0].balances == {"USDT": 9_999.0}
