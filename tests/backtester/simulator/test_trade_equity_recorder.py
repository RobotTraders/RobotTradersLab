from datetime import datetime

import pytest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import (
    EquitySnapshot,
    SimulatedPosition,
)
from robottraderslab.backtester.simulator.trade_equity_recorder import (
    TradeEquityRecorder,
)
from robottraderslab.exchanges import Currency, PositionSide

BTC_USDT_PERP = Symbol.create("BTC/USDT:USDT")
ETH_USDT_PERP = Symbol.create("ETH/USDT:USDT")
ZERO_POINT_ONE_BTC_IN_STEPS = 10_000_000


@pytest.fixture
def trade_equity_recorder() -> TradeEquityRecorder:
    return TradeEquityRecorder()


@pytest.fixture
def balances() -> dict[Currency, float]:
    return {"USDT": 10_000.0}


@pytest.fixture
def sample_positions() -> dict[Symbol, SimulatedPosition]:
    btc_position = SimulatedPosition(
        symbol=BTC_USDT_PERP,
        side=PositionSide.LONG,
        leverage=1.0,
        taker_fee_rate=0.001,
        entry_time=datetime.fromisoformat("2024-01-01 10:00:00"),
    )
    btc_position.add_to_position(ZERO_POINT_ONE_BTC_IN_STEPS, 50_000.0)
    return {BTC_USDT_PERP: btc_position}


@pytest.fixture
def futures_prices() -> dict[Symbol, float]:
    return {BTC_USDT_PERP: 55_000.0, ETH_USDT_PERP: 3000.0}


class TestConstruction:
    def test_construction(self):
        trade_equity_recorder = TradeEquityRecorder()

        assert trade_equity_recorder.snapshots == []


class TestTradeEquityRecording:
    def test_records_snapshot(
        self, trade_equity_recorder, balances, sample_positions, futures_prices
    ):
        timestamp = datetime.fromisoformat("2024-01-01 14:30:00")
        snapshot = EquitySnapshot(
            timestamp=timestamp,
            market_type="futures",
            balances=balances,
            positions=sample_positions,
            prices=futures_prices,
        )

        trade_equity_recorder.record_snapshot(snapshot)

        assert len(trade_equity_recorder.snapshots) == 1
        assert trade_equity_recorder.snapshots[0].timestamp == timestamp
        assert trade_equity_recorder.snapshots[0].market_type == "futures"
        assert trade_equity_recorder.snapshots[0].balances == balances
        assert trade_equity_recorder.snapshots[0].positions == sample_positions
        assert trade_equity_recorder.snapshots[0].prices == futures_prices

    def test_multiple_trade_snapshots(
        self, trade_equity_recorder, balances, sample_positions, futures_prices
    ):
        timestamps = [
            datetime.fromisoformat("2024-01-01 09:15:00"),
            datetime.fromisoformat("2024-01-01 14:30:00"),
            datetime.fromisoformat("2024-01-02 16:45:00"),
        ]

        for timestamp in timestamps:
            snapshot = EquitySnapshot(
                timestamp=timestamp,
                market_type="futures",
                balances=balances,
                positions=sample_positions,
                prices=futures_prices,
            )
            trade_equity_recorder.record_snapshot(snapshot)

        assert len(trade_equity_recorder.snapshots) == 3
        for i, snapshot in enumerate(trade_equity_recorder.snapshots):
            assert snapshot.timestamp == timestamps[i]
