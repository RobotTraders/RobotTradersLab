import logging
import re
from datetime import datetime
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from robottraderslab import BotConfig, Symbol
from robottraderslab._core import OHLCVProviderProtocol
from robottraderslab.backtester import BacktestOutputs
from robottraderslab.backtester.simulator import (
    EquitySnapshot,
    MissingConversionRateError,
    SimulatedPosition,
)
from robottraderslab.exchanges import Balance, Currency, PositionSide

BTC_USDT = Symbol.create("BTC/USDT")
ETH_USDT = Symbol.create("ETH/USDT")
BTC_USDT_PERP = Symbol.create("BTC/USDT:USDT")
ETH_USDT_PERP = Symbol.create("ETH/USDT:USDT")
ZERO_POINT_ONE_BTC_IN_STEPS = 10_000_000

REPORT_PREAMBLE = """
[strategy]
strategy_class = "impulse"

[report]
"""

PROFILE_CONFIG_TOML = """
[strategy]
strategy_class = "impulse"

[backtest]
initial_balance = { USDT = 1000.0 }
maker_fee_rate = 0.0
taker_fee_rate = 0.0
start_date = "2024-01-01"
end_date = "2024-01-02"
ohlcv_provider = {}

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1h"
tag = "alpha"
trix_length = 31
signal_length = 5

[[strategy.profiles]]
symbol = "BTC/USDT:USDT"
timeframe = "1h"
tag = "beta"
trix_length = 43
signal_length = 5
"""


@pytest.fixture
def futures_balances() -> dict[Currency, Balance]:
    return {"USDT": Balance(locked=0.0, total=10_000.0)}


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


class TestBalanceCurves:
    def test_empty_equity_snapshots(self):
        equity_snapshots = []
        backtest_outputs = BacktestOutputs(
            {},
            "USDT",
            equity_snapshots,
            equity_snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )
        curves = backtest_outputs.get_balance_curves()

        assert isinstance(curves, pd.DataFrame)
        assert len(curves) == 0

    def test_single_equity_snapshots(
        self,
        futures_balances,
        sample_positions,
        futures_prices,
    ):
        timestamp = datetime.fromisoformat("2024-01-01 00:00:00")
        snapshots = [
            EquitySnapshot(
                timestamp=timestamp,
                market_type="futures",
                balances=futures_balances,
                positions=sample_positions,
                prices=futures_prices,
            )
        ]

        backtest_outputs = BacktestOutputs(
            futures_balances,
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )
        curves = backtest_outputs.get_balance_curves()

        assert len(curves) == 1
        assert curves.index[0] == timestamp
        assert curves.index.name == "timestamp"
        assert "USDT" in curves.columns
        assert curves.loc[timestamp, "USDT"] == 10_000.0

    def test_multiple_equity_snapshots_same_currencies(
        self, sample_positions, futures_prices
    ):
        timestamps = [
            datetime.fromisoformat("2024-01-01 00:00:00"),
            datetime.fromisoformat("2024-01-02 00:00:00"),
        ]
        balances = [
            {"USDT": Balance(locked=0.0, total=10_000.0)},
            {"USDT": Balance(locked=0.0, total=12_000.0)},
        ]

        snapshots = list()
        for timestamp, balance in zip(timestamps, balances):
            snapshots.append(
                EquitySnapshot(
                    timestamp=timestamp,
                    market_type="futures",
                    balances=balance,
                    positions=sample_positions,
                    prices=futures_prices,
                )
            )

        backtest_outputs = BacktestOutputs(
            balances[-1],
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )
        curves = backtest_outputs.get_balance_curves()

        assert len(curves) == 2
        assert curves.loc[timestamps[0], "USDT"] == 10_000.0
        assert curves.loc[timestamps[1], "USDT"] == 12_000.0

    def test_multiple_currencies(self, sample_positions, futures_prices):
        timestamp = datetime.fromisoformat("2024-01-01 00:00:00")
        balances = {
            "USDT": Balance(locked=0.0, total=10_000.0),
            "BTC": Balance(locked=0.0, total=0.5),
        }

        snapshots = [
            EquitySnapshot(
                timestamp=timestamp,
                market_type="futures",
                balances=balances,
                positions=sample_positions,
                prices=futures_prices,
            )
        ]

        backtest_outputs = BacktestOutputs(
            balances,
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )
        curves = backtest_outputs.get_balance_curves()

        assert len(curves) == 1
        assert "USDT" in curves.columns
        assert "BTC" in curves.columns
        assert curves.loc[timestamp, "USDT"] == 10_000.0
        assert curves.loc[timestamp, "BTC"] == 0.5

    def test_spot_multi_asset_balance_curves(self):
        timestamps = [
            datetime.fromisoformat("2024-01-01 00:00:00"),
            datetime.fromisoformat("2024-01-02 00:00:00"),
        ]
        # Simulate buying BTC with USDT
        balances = [
            {
                "USDT": Balance(locked=0.0, total=10_000.0),
                "BTC": Balance(locked=0.0, total=0.0),
            },
            {
                "USDT": Balance(locked=0.0, total=5_000.0),
                "BTC": Balance(locked=0.0, total=0.1),
            },
        ]
        prices = {BTC_USDT: 50_000.0}

        snapshots = [
            EquitySnapshot(
                timestamp=ts,
                market_type="spot",
                balances=b,
                positions={},
                prices=prices,
            )
            for ts, b in zip(timestamps, balances)
        ]

        backtest_outputs = BacktestOutputs(
            balances[-1],
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )
        curves = backtest_outputs.get_balance_curves()

        assert len(curves) == 2
        assert "USDT" in curves.columns
        assert "BTC" in curves.columns
        assert curves.loc[timestamps[0], "USDT"] == 10_000.0
        assert curves.loc[timestamps[0], "BTC"] == 0.0
        assert curves.loc[timestamps[1], "USDT"] == 5_000.0
        assert curves.loc[timestamps[1], "BTC"] == 0.1

    def test_missing_currency_in_later_snapshot(self, sample_positions, futures_prices):
        timestamps = [
            datetime.fromisoformat("2024-01-01 00:00:00"),
            datetime.fromisoformat("2024-01-02 00:00:00"),
        ]
        balances = [
            {
                "USDT": Balance(locked=0.0, total=10_000.0),
                "BTC": Balance(locked=0.0, total=0.5),
            },
            {
                "USDT": Balance(locked=0.0, total=12_000.0),
            },  # BTC missing
        ]

        snapshots = list()
        for timestamp, balance in zip(timestamps, balances):
            snapshots.append(
                EquitySnapshot(
                    timestamp=timestamp,
                    market_type="futures",
                    balances=balance,
                    positions=sample_positions,
                    prices=futures_prices,
                )
            )

        backtest_outputs = BacktestOutputs(
            balances[-1],
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )
        curves = backtest_outputs.get_balance_curves()

        assert len(curves) == 2
        assert curves.loc[timestamps[0], "BTC"] == 0.5
        assert (
            curves.loc[timestamps[1], "BTC"] == 0.0
        )  # Missing currency defaults to 0.0


class TestEquityCurveCalculation:
    def test_empty_equity_snapshots(self):
        equity_snapshots = []
        backtest_outputs = BacktestOutputs(
            {},
            "USDT",
            equity_snapshots,
            equity_snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )
        equity_curve = backtest_outputs.get_equity_curve()

        assert isinstance(equity_curve, pd.Series)
        assert len(equity_curve) == 0
        assert equity_curve.dtype == float

    def test_futures_single_snapshot(
        self,
        futures_balances,
        sample_positions,
        futures_prices,
    ):
        timestamp = datetime.fromisoformat("2024-01-01 00:00:00")
        snapshots = [
            EquitySnapshot(
                timestamp=timestamp,
                market_type="futures",
                balances=futures_balances,
                positions=sample_positions,
                prices=futures_prices,
            )
        ]

        backtest_outputs = BacktestOutputs(
            futures_balances,
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )
        equity_curve = backtest_outputs.get_equity_curve()

        assert len(equity_curve) == 1
        assert equity_curve.index[0] == timestamp
        expected_equity = 10_000.0 + 500.0
        assert equity_curve.iloc[0] == expected_equity

    def test_futures_multiple_snapshots(self, futures_balances, sample_positions):
        timestamps = [
            datetime.fromisoformat("2024-01-01 00:00:00"),
            datetime.fromisoformat("2024-01-02 00:00:00"),
        ]
        prices = [
            {BTC_USDT_PERP: 55_000.0},
            {BTC_USDT_PERP: 60_000.0},
        ]

        snapshots = list()
        for timestamp, price in zip(timestamps, prices):
            snapshots.append(
                EquitySnapshot(
                    timestamp=timestamp,
                    market_type="futures",
                    balances=futures_balances,
                    positions=sample_positions,
                    prices=price,
                )
            )

        backtest_outputs = BacktestOutputs(
            futures_balances,
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )
        equity_curve = backtest_outputs.get_equity_curve()

        assert len(equity_curve) == 2
        assert equity_curve.index[0] == timestamps[0]
        assert equity_curve.index[1] == timestamps[1]
        assert equity_curve.iloc[0] == 10_000.0 + 500.0
        assert equity_curve.iloc[1] == 10_000.0 + 1000.0

    def test_spot_single_snapshot(self):
        timestamp = datetime.fromisoformat("2024-01-01 00:00:00")
        spot_balances = {
            "USDT": Balance(locked=0.0, total=5000.0),
            "BTC": Balance(locked=0.0, total=0.1),
            "ETH": Balance(locked=0.0, total=2.0),
        }
        empty_positions = {}
        spot_prices = {BTC_USDT: 50_000.0, ETH_USDT: 3000.0}

        snapshots = [
            EquitySnapshot(
                timestamp=timestamp,
                market_type="spot",
                balances=spot_balances,
                positions=empty_positions,
                prices=spot_prices,
            )
        ]

        backtest_outputs = BacktestOutputs(
            spot_balances,
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )
        equity_curve = backtest_outputs.get_equity_curve()

        assert len(equity_curve) == 1
        expected_equity = 5000.0 + (0.1 * 50_000.0) + (2.0 * 3000.0)
        assert equity_curve.iloc[0] == expected_equity

    def test_spot_multiple_snapshots_equity_tracks_price_changes(self):
        timestamps = [
            datetime.fromisoformat("2024-01-01 00:00:00"),
            datetime.fromisoformat("2024-01-02 00:00:00"),
            datetime.fromisoformat("2024-01-03 00:00:00"),
        ]
        balances = {
            "USDT": Balance(locked=0.0, total=5_000.0),
            "BTC": Balance(locked=0.0, total=0.1),
            "ETH": Balance(locked=0.0, total=2.0),
        }
        prices = [
            {BTC_USDT: 50_000.0, ETH_USDT: 3_000.0},
            {BTC_USDT: 55_000.0, ETH_USDT: 3_500.0},
            {BTC_USDT: 45_000.0, ETH_USDT: 2_500.0},
        ]

        snapshots = [
            EquitySnapshot(
                timestamp=ts,
                market_type="spot",
                balances=balances,
                positions={},
                prices=p,
            )
            for ts, p in zip(timestamps, prices)
        ]

        backtest_outputs = BacktestOutputs(
            balances,
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )
        equity_curve = backtest_outputs.get_equity_curve()

        assert len(equity_curve) == 3
        # Day 1: 5000 + 0.1*50000 + 2*3000 = 5000 + 5000 + 6000 = 16000
        assert equity_curve.iloc[0] == 16_000.0
        # Day 2: 5000 + 0.1*55000 + 2*3500 = 5000 + 5500 + 7000 = 17500
        assert equity_curve.iloc[1] == 17_500.0
        # Day 3: 5000 + 0.1*45000 + 2*2500 = 5000 + 4500 + 5000 = 14500
        assert equity_curve.iloc[2] == 14_500.0

    def test_spot_multiple_snapshots_equity_tracks_balance_changes(self):
        timestamps = [
            datetime.fromisoformat("2024-01-01 00:00:00"),
            datetime.fromisoformat("2024-01-02 00:00:00"),
        ]
        # Simulate buying BTC at 50000, then price rises to 52000
        balances = [
            {
                "USDT": Balance(locked=0.0, total=10_000.0),
                "BTC": Balance(locked=0.0, total=0.0),
            },
            {
                "USDT": Balance(locked=0.0, total=5_000.0),
                "BTC": Balance(locked=0.0, total=0.1),
            },
        ]
        prices = [
            {BTC_USDT: 50_000.0},
            {BTC_USDT: 52_000.0},
        ]

        snapshots = [
            EquitySnapshot(
                timestamp=ts,
                market_type="spot",
                balances=b,
                positions={},
                prices=p,
            )
            for ts, b, p in zip(timestamps, balances, prices)
        ]

        backtest_outputs = BacktestOutputs(
            balances[-1],
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )
        equity_curve = backtest_outputs.get_equity_curve()

        assert len(equity_curve) == 2
        # Before buy: 10000 USDT + 0 BTC * 50000 = 10000
        assert equity_curve.iloc[0] == 10_000.0
        # After buy + price rise: 5000 USDT + 0.1 BTC * 52000 = 5000 + 5200 = 10200
        assert equity_curve.iloc[1] == 10_200.0


class TestEdgeCases:
    def test_futures_equity_with_no_positions(self):
        timestamp = datetime.fromisoformat("2024-01-01 00:00:00")
        balances = {"USDT": Balance(locked=0.0, total=10_000.0)}

        empty_positions = {}
        prices = {BTC_USDT_PERP: 50_000.0}

        snapshots = [
            EquitySnapshot(
                timestamp=timestamp,
                market_type="futures",
                balances=balances,
                positions=empty_positions,
                prices=prices,
            )
        ]

        backtest_outputs = BacktestOutputs(
            balances,
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )
        equity_curve = backtest_outputs.get_equity_curve()

        assert len(equity_curve) == 1
        assert equity_curve.iloc[0] == 10_000.0

    def test_futures_equity_with_position_but_no_matching_price(self, sample_positions):
        timestamp = datetime.fromisoformat("2024-01-01 00:00:00")
        balances = {"USDT": Balance(locked=0.0, total=10_000.0)}
        prices = {ETH_USDT_PERP: 3000.0}  # No BTC price

        snapshots = [
            EquitySnapshot(
                timestamp=timestamp,
                market_type="futures",
                balances=balances,
                positions=sample_positions,
                prices=prices,
            )
        ]

        backtest_outputs = BacktestOutputs(
            balances,
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )

        with pytest.raises(KeyError):
            backtest_outputs.get_equity_curve()

    def test_spot_equity_with_no_matching_price(self):
        timestamp = datetime.fromisoformat("2024-01-01 00:00:00")
        balances = {
            "USDT": Balance(locked=0.0, total=10_000.0),
            "BTC": Balance(locked=0.0, total=0.1),
        }
        prices = {ETH_USDT: 3000.0}  # No BTC price
        empty_positions = {}

        snapshots = [
            EquitySnapshot(
                timestamp=timestamp,
                market_type="spot",
                balances=balances,
                positions=empty_positions,
                prices=prices,
            )
        ]

        backtest_outputs = BacktestOutputs(
            balances,
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )

        with pytest.raises(MissingConversionRateError):
            backtest_outputs.get_equity_curve()

    def test_zero_balance_and_positions(self):
        timestamp = datetime.fromisoformat("2024-01-01 00:00:00")
        balances = {"USDT": Balance(locked=0.0, total=0.0)}
        empty_positions = {}
        prices = {BTC_USDT_PERP: 50_000.0}

        snapshots = [
            EquitySnapshot(
                timestamp=timestamp,
                market_type="futures",
                balances=balances,
                positions=empty_positions,
                prices=prices,
            )
        ]

        backtest_outputs = BacktestOutputs(
            balances,
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )

        equity_curve = backtest_outputs.get_equity_curve()

        assert len(equity_curve) == 1
        assert equity_curve.iloc[0] == 0.0

    def test_negative_balance(self):
        timestamp = datetime.fromisoformat("2024-01-01 00:00:00")
        balances = {"USDT": Balance(locked=0.0, total=-1000.0)}
        empty_positions = {}
        prices = {BTC_USDT_PERP: 50_000.0}

        snapshots = [
            EquitySnapshot(
                timestamp=timestamp,
                market_type="futures",
                balances=balances,
                positions=empty_positions,
                prices=prices,
            )
        ]

        backtest_outputs = BacktestOutputs(
            balances,
            "USDT",
            snapshots,
            snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )

        equity_curve = backtest_outputs.get_equity_curve()

        assert len(equity_curve) == 1
        assert equity_curve.iloc[0] == -1000.0


class TestSourceParameterSelection:
    def test_get_equity_curve_uses_daily_by_default(
        self, futures_balances, sample_positions, futures_prices
    ):
        daily_snapshots = [
            EquitySnapshot(
                timestamp=datetime.fromisoformat("2024-01-01 00:00:00"),
                market_type="futures",
                balances=futures_balances,
                positions=sample_positions,
                prices=futures_prices,
            )
        ]
        trade_snapshots = [
            EquitySnapshot(
                timestamp=datetime.fromisoformat("2024-01-01 10:30:00"),
                market_type="futures",
                balances=futures_balances,
                positions=sample_positions,
                prices=futures_prices,
            )
        ]

        backtest_outputs = BacktestOutputs(
            futures_balances,
            "USDT",
            daily_snapshots,
            trade_snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )

        equity_curve = backtest_outputs.get_equity_curve()

        assert len(equity_curve) == 1
        assert equity_curve.index[0] == datetime.fromisoformat("2024-01-01 00:00:00")

    def test_get_equity_curve_with_daily_source(
        self, futures_balances, sample_positions, futures_prices
    ):
        daily_snapshots = [
            EquitySnapshot(
                timestamp=datetime.fromisoformat("2024-01-01 00:00:00"),
                market_type="futures",
                balances=futures_balances,
                positions=sample_positions,
                prices=futures_prices,
            )
        ]
        trade_snapshots = [
            EquitySnapshot(
                timestamp=datetime.fromisoformat("2024-01-01 10:30:00"),
                market_type="futures",
                balances=futures_balances,
                positions=sample_positions,
                prices=futures_prices,
            )
        ]

        backtest_outputs = BacktestOutputs(
            futures_balances,
            "USDT",
            daily_snapshots,
            trade_snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )

        equity_curve = backtest_outputs.get_equity_curve(source="daily")

        assert len(equity_curve) == 1
        assert equity_curve.index[0] == datetime.fromisoformat("2024-01-01 00:00:00")

    def test_get_equity_curve_with_trade_source(
        self, futures_balances, sample_positions, futures_prices
    ):
        daily_snapshots = [
            EquitySnapshot(
                timestamp=datetime.fromisoformat("2024-01-01 00:00:00"),
                market_type="futures",
                balances=futures_balances,
                positions=sample_positions,
                prices=futures_prices,
            )
        ]
        trade_snapshots = [
            EquitySnapshot(
                timestamp=datetime.fromisoformat("2024-01-01 10:30:00"),
                market_type="futures",
                balances=futures_balances,
                positions=sample_positions,
                prices=futures_prices,
            )
        ]

        backtest_outputs = BacktestOutputs(
            futures_balances,
            "USDT",
            daily_snapshots,
            trade_snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )

        equity_curve = backtest_outputs.get_equity_curve(source="trade")

        assert len(equity_curve) == 1
        assert equity_curve.index[0] == datetime.fromisoformat("2024-01-01 10:30:00")

    def test_get_balance_curves_uses_daily_by_default(
        self, futures_balances, sample_positions, futures_prices
    ):
        daily_snapshots = [
            EquitySnapshot(
                timestamp=datetime.fromisoformat("2024-01-01 00:00:00"),
                market_type="futures",
                balances=futures_balances,
                positions=sample_positions,
                prices=futures_prices,
            )
        ]
        trade_snapshots = [
            EquitySnapshot(
                timestamp=datetime.fromisoformat("2024-01-01 10:30:00"),
                market_type="futures",
                balances=futures_balances,
                positions=sample_positions,
                prices=futures_prices,
            )
        ]

        backtest_outputs = BacktestOutputs(
            futures_balances,
            "USDT",
            daily_snapshots,
            trade_snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )

        balance_curves = backtest_outputs.get_balance_curves()

        assert len(balance_curves) == 1
        assert balance_curves.index[0] == datetime.fromisoformat("2024-01-01 00:00:00")

    def test_get_balance_curves_with_trade_source(
        self, futures_balances, sample_positions, futures_prices
    ):
        daily_snapshots = [
            EquitySnapshot(
                timestamp=datetime.fromisoformat("2024-01-01 00:00:00"),
                market_type="futures",
                balances=futures_balances,
                positions=sample_positions,
                prices=futures_prices,
            )
        ]
        trade_snapshots = [
            EquitySnapshot(
                timestamp=datetime.fromisoformat("2024-01-01 10:30:00"),
                market_type="futures",
                balances=futures_balances,
                positions=sample_positions,
                prices=futures_prices,
            )
        ]

        backtest_outputs = BacktestOutputs(
            futures_balances,
            "USDT",
            daily_snapshots,
            trade_snapshots,
            pd.DataFrame(),
            cast(OHLCVProviderProtocol, None),
        )

        balance_curves = backtest_outputs.get_balance_curves(source="trade")

        assert len(balance_curves) == 1
        assert balance_curves.index[0] == datetime.fromisoformat("2024-01-01 10:30:00")


@pytest.fixture
def fills_with_complete_trade() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTC/USDT:USDT", "BTC/USDT:USDT"],
            "side": ["long", "long"],
            "gross_quantity": [1.0, 1.0],
            "net_quantity": [0.999, 0.999],
            "price": [45_000.0, 46_000.0],
            "fee": [45.0, 46.0],
            "fill_type": ["enter_long", "exit_long"],
        },
        index=pd.DatetimeIndex(
            [
                datetime.fromisoformat("2024-01-01 10:00:00"),
                datetime.fromisoformat("2024-01-02 10:00:00"),
            ],
        ),
    )


@pytest.fixture
def equity_snapshots() -> list[EquitySnapshot]:
    balances = {"USDT": Balance(locked=0.0, total=10_000.0)}
    return [
        EquitySnapshot(
            timestamp=datetime.fromisoformat("2024-01-01 00:00:00"),
            market_type="futures",
            balances=balances,
            positions={},
            prices={BTC_USDT_PERP: 50_000.0},
        ),
        EquitySnapshot(
            timestamp=datetime.fromisoformat("2024-01-02 00:00:00"),
            market_type="futures",
            balances=balances,
            positions={},
            prices={BTC_USDT_PERP: 51_000.0},
        ),
    ]


class TestCreateAnalyser:
    def test_with_complete_trade(self, fills_with_complete_trade, equity_snapshots):
        from robottraderslab.analyser import Analyser

        ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
        outputs = BacktestOutputs(
            final_balance={"USDT": 10_900.0},
            equity_currency="USDT",
            daily_equity_snapshots=equity_snapshots,
            trade_equity_snapshots=equity_snapshots,
            fills=fills_with_complete_trade,
            ohlcv_provider=ohlcv_provider,
        )

        analyser = outputs.create_analyser()

        assert isinstance(analyser, Analyser)

    def test_with_no_trades(self, equity_snapshots):
        empty_fills = pd.DataFrame(
            columns=[
                "symbol",
                "side",
                "gross_quantity",
                "net_quantity",
                "price",
                "fee",
                "fill_type",
            ]
        )
        outputs = BacktestOutputs(
            final_balance={"USDT": 10_000.0},
            equity_currency="USDT",
            daily_equity_snapshots=equity_snapshots,
            trade_equity_snapshots=equity_snapshots,
            fills=empty_fills,
            ohlcv_provider=cast(OHLCVProviderProtocol, None),
        )

        with pytest.raises(RuntimeError, match="No trades generated"):
            outputs.create_analyser()

    @pytest.fixture
    def fills_with_intra_candle_round_trips(self) -> pd.DataFrame:
        """Many symbols each opening and closing within the same candle.

        Every fill shares one timestamp, with each symbol's entry recorded
        before its exit. This is the pattern a mean-reversion strategy produces
        on a volatile candle, and it must survive the analyser's trade
        aggregation without losing the entry-before-exit order.
        """
        ts = datetime.fromisoformat("2024-01-02 10:00:00")
        symbols = [
            "UNI/USDT:USDT",
            "ADA/USDT:USDT",
            "DOGE/USDT:USDT",
            "DOT/USDT:USDT",
            "LINK/USDT:USDT",
            "LTC/USDT:USDT",
            "NEAR/USDT:USDT",
            "SHIB/USDT:USDT",
            "AVAX/USDT:USDT",
            "PENDLE/USDT:USDT",
        ]
        rows = []
        index = []
        for symbol in symbols:
            rows.append((symbol, "long", 1.0, 0.999, 100.0, 0.1, "enter_long"))
            index.append(ts)
            rows.append((symbol, "long", 0.999, 0.999, 105.0, 0.1, "exit_long"))
            index.append(ts)
        return pd.DataFrame(
            rows,
            columns=[
                "symbol",
                "side",
                "gross_quantity",
                "net_quantity",
                "price",
                "fee",
                "fill_type",
            ],
            index=pd.DatetimeIndex(index),
        )

    def test_with_intra_candle_round_trips(
        self, fills_with_intra_candle_round_trips, equity_snapshots
    ):
        from robottraderslab.analyser import Analyser

        ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
        outputs = BacktestOutputs(
            final_balance={"USDT": 10_000.0},
            equity_currency="USDT",
            daily_equity_snapshots=equity_snapshots,
            trade_equity_snapshots=equity_snapshots,
            fills=fills_with_intra_candle_round_trips,
            ohlcv_provider=ohlcv_provider,
        )

        analyser = outputs.create_analyser()

        assert isinstance(analyser, Analyser)
        assert len(analyser.trades) == 10

    def test_profiles_declared_in_the_config_reach_the_chart_service(
        self, fills_with_complete_trade, equity_snapshots
    ):
        outputs = BacktestOutputs(
            final_balance={"USDT": 10_900.0},
            equity_currency="USDT",
            daily_equity_snapshots=equity_snapshots,
            trade_equity_snapshots=equity_snapshots,
            fills=fills_with_complete_trade,
            ohlcv_provider=Mock(spec=OHLCVProviderProtocol),
            bot_config=BotConfig.from_text(PROFILE_CONFIG_TOML),
        )
        analyser = outputs.create_analyser()

        with patch.object(analyser, "_chart_service") as chart_service:
            analyser.plot_candlesticks(indicators_name="impulse")

        charted = chart_service.open_candlesticks.call_args[1]["charts"]
        assert [chart["tag"] for chart in charted] == ["alpha", "beta"]

    def test_the_reference_the_report_section_declares(
        self, fills_with_complete_trade, equity_snapshots
    ):
        ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
        ohlcv_provider.fetch_ohlcv.return_value = pd.DataFrame(
            {
                "open": [50_000.0, 51_000.0],
                "high": [51_000.0, 52_000.0],
                "low": [49_000.0, 50_000.0],
                "close": [50_500.0, 51_500.0],
                "volume": [1000.0, 1100.0],
            },
            index=pd.DatetimeIndex(
                [
                    datetime.fromisoformat("2024-01-01 00:00:00"),
                    datetime.fromisoformat("2024-01-02 00:00:00"),
                ],
            ),
        )
        outputs = BacktestOutputs(
            final_balance={"USDT": 10_900.0},
            equity_currency="USDT",
            daily_equity_snapshots=equity_snapshots,
            trade_equity_snapshots=equity_snapshots,
            fills=fills_with_complete_trade,
            ohlcv_provider=ohlcv_provider,
            bot_config=BotConfig.from_text(
                REPORT_PREAMBLE + f'reference_symbol = "{BTC_USDT_PERP}"'
            ),
        )

        analyser = outputs.create_analyser()

        ohlcv_provider.fetch_ohlcv.assert_called_once_with(BTC_USDT_PERP, "1d")
        assert analyser.reference_price is not None


class TestSavedRunFolder:
    @pytest.fixture
    def bot_config(self, tmp_path: Path) -> BotConfig:
        toml_file = tmp_path / "impulse-bot-example.toml"
        toml_file.write_bytes(PROFILE_CONFIG_TOML.encode("utf-8"))
        return BotConfig.from_file(toml_file)

    @pytest.fixture
    def ohlcv_provider(self) -> Mock:
        provider = Mock(spec=OHLCVProviderProtocol)
        provider.fetch_ohlcv.return_value = pd.DataFrame(
            {
                "open": [50_000.0, 51_000.0],
                "high": [51_000.0, 52_000.0],
                "low": [49_000.0, 50_000.0],
                "close": [50_500.0, 51_500.0],
                "volume": [1000.0, 1100.0],
            },
            index=pd.DatetimeIndex(
                [
                    datetime.fromisoformat("2024-01-01 00:00:00"),
                    datetime.fromisoformat("2024-01-02 00:00:00"),
                ],
            ),
        )
        return provider

    @pytest.fixture(autouse=True)
    def _no_chart_lines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The saved run's own indicator maths are the impulse plugin's to
        test; here only the folder and the page around them are exercised.
        """
        monkeypatch.setattr(
            "robottraderslab.analyser.analyser.load_lightweight_chart_indicators",
            lambda indicators_name, candles, params: [],
        )

    @pytest.fixture
    def outputs(
        self,
        fills_with_complete_trade: pd.DataFrame,
        equity_snapshots: list[EquitySnapshot],
        bot_config: BotConfig,
        ohlcv_provider: Mock,
    ) -> BacktestOutputs:
        return BacktestOutputs(
            final_balance={"USDT": 10_900.0},
            equity_currency="USDT",
            daily_equity_snapshots=equity_snapshots,
            trade_equity_snapshots=equity_snapshots,
            fills=fills_with_complete_trade,
            ohlcv_provider=ohlcv_provider,
            bot_config=bot_config,
        )

    def _run_folder(self, bot_config: BotConfig) -> Path:
        reports_root = cast(Path, bot_config.config_dir) / "reports"
        [folder] = list(reports_root.iterdir())
        return folder

    def test_a_saved_run_writes_its_configuration_and_report(self, outputs, bot_config):
        outputs.create_analyser()

        run_folder = self._run_folder(bot_config)
        assert re.match(
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_backtest_impulse-bot-example",
            run_folder.name,
        )
        assert (run_folder / "impulse-bot-example.toml").read_bytes() == cast(
            Path, bot_config.config_file
        ).read_bytes()
        assert (run_folder / "report.txt").exists()

    def test_a_saved_run_writes_one_combined_page(self, outputs, bot_config):
        outputs.create_analyser()

        html_files = sorted((self._run_folder(bot_config)).glob("*.html"))
        assert [path.name for path in html_files] == ["report.html"]

    def test_the_saved_page_embeds_every_profile(
        self, outputs, bot_config, read_payload
    ):
        outputs.create_analyser()

        report_html = self._run_folder(bot_config) / "report.html"
        labels = [market["label"] for market in read_payload(report_html)["markets"]]
        assert labels == ["BTC/USDT:USDT · 1h · alpha", "BTC/USDT:USDT · 1h · beta"]

    def test_the_saved_page_carries_the_configuration_and_a_run_titled_page(
        self, outputs, bot_config, read_payload
    ):
        outputs.create_analyser()

        report_html = self._run_folder(bot_config) / "report.html"
        drawn = read_payload(report_html)
        assert drawn["configuration"] == cast(Path, bot_config.config_file).read_text(
            encoding="utf-8"
        )
        assert drawn["title"].startswith("impulse-bot-example")

    def test_save_false_leaves_no_reports_directory(self, outputs, bot_config):
        outputs.create_analyser(save=False)

        assert not (cast(Path, bot_config.config_dir) / "reports").exists()

    def test_plot_candlesticks_on_a_saved_run_opens_without_rendering_again(
        self, outputs, bot_config, browser_openings
    ):
        analyser = outputs.create_analyser()
        report_html = self._run_folder(bot_config) / "report.html"

        with patch.object(
            analyser, "_chart_service", wraps=analyser._chart_service
        ) as chart_service:
            written = analyser.plot_candlesticks(indicators_name="impulse")

        chart_service.write_report.assert_not_called()
        assert written == [report_html]
        assert browser_openings[-1] == report_html.as_uri()

    def test_plot_candlesticks_on_an_unsaved_run_renders_fresh(
        self, outputs, bot_config, browser_openings
    ):
        analyser = outputs.create_analyser(save=False)

        written = analyser.plot_candlesticks(indicators_name="impulse")

        assert not (cast(Path, bot_config.config_dir) / "reports").exists()
        assert len(written) == 2

    def test_a_chart_failure_still_saves_the_configuration_and_report(
        self,
        outputs,
        bot_config,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        monkeypatch.setattr(
            "robottraderslab.analyser.analyser.load_lightweight_chart_indicators",
            Mock(side_effect=RuntimeError("boom")),
        )

        with caplog.at_level(logging.WARNING):
            outputs.create_analyser()

        run_folder = self._run_folder(bot_config)
        assert (run_folder / "impulse-bot-example.toml").exists()
        assert (run_folder / "report.txt").exists()
        assert not list(run_folder.glob("*.html"))
        assert "No chart page written" in caplog.text
