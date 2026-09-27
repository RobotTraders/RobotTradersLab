from datetime import datetime

import pytest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import (
    EquitySnapshot,
    MissingConversionRateError,
    SimulatedPosition,
)
from robottraderslab.backtester.simulator.equity_snapshot import Prices
from robottraderslab.exchanges import Balance, Currency, PositionSide

BTC_USDT = Symbol.create("BTC/USDT")
ETH_USDT = Symbol.create("ETH/USDT")
BTC_USDT_PERP = Symbol.create("BTC/USDT:USDT")
ETH_USDT_PERP = Symbol.create("ETH/USDT:USDT")

TIMESTAMP = datetime.fromisoformat("2024-01-01 00:00:00")
ZERO_POINT_ONE_BTC_IN_STEPS = 10_000_000


def test_frozen_copy_creates_independent_snapshot():
    balances: dict[Currency, float] = {}
    positions: dict[Symbol, SimulatedPosition] = {}
    prices: Prices = {}

    snapshot = EquitySnapshot(TIMESTAMP, "futures", balances, positions, prices)
    copied = snapshot.frozen_copy()

    assert copied.balances is not snapshot.balances
    assert copied.positions is not snapshot.positions
    assert copied.prices is not snapshot.prices


class TestFuturesEquity:
    def test_equity_only_balance_no_positions(self):
        balances = {"USDT": Balance(locked=0.0, total=10_000.0)}
        snapshot = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="futures",
            balances=balances,
            positions={},
            prices={BTC_USDT_PERP: 50_000.0},
        )

        assert snapshot.get_equity("USDT") == 10_000.0

    def test_equity_with_long_position_at_profit(self):
        btc_position = SimulatedPosition(
            symbol=BTC_USDT_PERP,
            side=PositionSide.LONG,
            leverage=1.0,
            taker_fee_rate=0.0,
            entry_time=TIMESTAMP,
        )
        btc_position.add_to_position(ZERO_POINT_ONE_BTC_IN_STEPS, 50_000.0)

        balances = {"USDT": Balance(locked=0.0, total=10_000.0)}
        snapshot = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="futures",
            balances=balances,
            positions={BTC_USDT_PERP: btc_position},
            prices={BTC_USDT_PERP: 60_000.0},
        )

        # 0.1 BTC position: entry 50000, current 60000 → unrealised PnL = 0.1 * (60000 - 50000) = 1000
        assert snapshot.get_equity("USDT") == 11_000.0

    def test_equity_with_long_position_at_loss(self):
        btc_position = SimulatedPosition(
            symbol=BTC_USDT_PERP,
            side=PositionSide.LONG,
            leverage=1.0,
            taker_fee_rate=0.0,
            entry_time=TIMESTAMP,
        )
        btc_position.add_to_position(ZERO_POINT_ONE_BTC_IN_STEPS, 50_000.0)

        balances = {"USDT": Balance(locked=0.0, total=10_000.0)}
        snapshot = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="futures",
            balances=balances,
            positions={BTC_USDT_PERP: btc_position},
            prices={BTC_USDT_PERP: 40_000.0},
        )

        # 0.1 BTC position: entry 50000, current 40000 → unrealised PnL = 0.1 * (40000 - 50000) = -1000
        assert snapshot.get_equity("USDT") == 9_000.0

    def test_equity_with_short_position_at_profit(self):
        btc_position = SimulatedPosition(
            symbol=BTC_USDT_PERP,
            side=PositionSide.SHORT,
            leverage=1.0,
            taker_fee_rate=0.0,
            entry_time=TIMESTAMP,
        )
        btc_position.add_to_position(ZERO_POINT_ONE_BTC_IN_STEPS, 50_000.0)

        balances = {"USDT": Balance(locked=0.0, total=10_000.0)}
        snapshot = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="futures",
            balances=balances,
            positions={BTC_USDT_PERP: btc_position},
            prices={BTC_USDT_PERP: 40_000.0},
        )

        # 0.1 BTC short: entry 50000, current 40000 → unrealised PnL = 0.1 * (50000 - 40000) = 1000
        assert snapshot.get_equity("USDT") == 11_000.0

    def test_equity_with_short_position_at_loss(self):
        btc_position = SimulatedPosition(
            symbol=BTC_USDT_PERP,
            side=PositionSide.SHORT,
            leverage=1.0,
            taker_fee_rate=0.0,
            entry_time=TIMESTAMP,
        )
        btc_position.add_to_position(ZERO_POINT_ONE_BTC_IN_STEPS, 50_000.0)

        balances = {"USDT": Balance(locked=0.0, total=10_000.0)}
        snapshot = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="futures",
            balances=balances,
            positions={BTC_USDT_PERP: btc_position},
            prices={BTC_USDT_PERP: 60_000.0},
        )

        # 0.1 BTC short: entry 50000, current 60000 → unrealised PnL = 0.1 * (50000 - 60000) = -1000
        assert snapshot.get_equity("USDT") == 9_000.0

    def test_equity_includes_locked_balance(self):
        # Locked USDT (e.g. margin reserved for an open order) counts toward equity
        balances = {"USDT": Balance(locked=2_000.0, total=10_000.0)}
        snapshot = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="futures",
            balances=balances,
            positions={},
            prices={},
        )

        assert snapshot.get_equity("USDT") == 10_000.0

    def test_equity_with_position_and_partial_locked_balance(self):
        btc_position = SimulatedPosition(
            symbol=BTC_USDT_PERP,
            side=PositionSide.LONG,
            leverage=1.0,
            taker_fee_rate=0.0,
            entry_time=TIMESTAMP,
        )
        btc_position.add_to_position(ZERO_POINT_ONE_BTC_IN_STEPS, 50_000.0)

        # 5000 locked as margin, 5000 available
        balances = {"USDT": Balance(locked=5_000.0, total=10_000.0)}
        snapshot = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="futures",
            balances=balances,
            positions={BTC_USDT_PERP: btc_position},
            prices={BTC_USDT_PERP: 60_000.0},
        )

        # equity = total (10000) + unrealised PnL (0.1 * 10000 = 1000)
        assert snapshot.get_equity("USDT") == 11_000.0

    def test_equity_missing_price_raises_key_error(self):
        btc_position = SimulatedPosition(
            symbol=BTC_USDT_PERP,
            side=PositionSide.LONG,
            leverage=1.0,
            taker_fee_rate=0.0,
            entry_time=TIMESTAMP,
        )
        btc_position.add_to_position(ZERO_POINT_ONE_BTC_IN_STEPS, 50_000.0)

        balances = {"USDT": Balance(locked=0.0, total=10_000.0)}
        snapshot = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="futures",
            balances=balances,
            positions={BTC_USDT_PERP: btc_position},
            prices={ETH_USDT_PERP: 3_000.0},  # BTC price missing
        )

        with pytest.raises(KeyError):
            snapshot.get_equity("USDT")


class TestSpotEquity:
    def test_equity_quote_currency_only(self):
        balances = {"USDT": Balance(locked=0.0, total=10_000.0)}
        snapshot = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="spot",
            balances=balances,
            positions={},
            prices={BTC_USDT: 50_000.0},
        )

        assert snapshot.get_equity("USDT") == 10_000.0

    def test_equity_single_base_asset(self):
        balances = {
            "USDT": Balance(locked=0.0, total=5_000.0),
            "BTC": Balance(locked=0.0, total=0.1),
        }
        snapshot = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="spot",
            balances=balances,
            positions={},
            prices={BTC_USDT: 50_000.0},
        )

        # 5000 USDT + 0.1 BTC * 50000 = 5000 + 5000 = 10000
        assert snapshot.get_equity("USDT") == 10_000.0

    def test_equity_multi_asset_portfolio(self):
        balances = {
            "USDT": Balance(locked=0.0, total=5_000.0),
            "BTC": Balance(locked=0.0, total=0.1),
            "ETH": Balance(locked=0.0, total=2.0),
        }
        snapshot = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="spot",
            balances=balances,
            positions={},
            prices={BTC_USDT: 50_000.0, ETH_USDT: 3_000.0},
        )

        # 5000 USDT + 0.1 BTC * 50000 + 2.0 ETH * 3000 = 5000 + 5000 + 6000 = 16000
        assert snapshot.get_equity("USDT") == 16_000.0

    def test_equity_reflects_price_increase(self):
        balances = {
            "USDT": Balance(locked=0.0, total=5_000.0),
            "BTC": Balance(locked=0.0, total=0.1),
        }
        snapshot_before = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="spot",
            balances=balances,
            positions={},
            prices={BTC_USDT: 50_000.0},
        )
        snapshot_after = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="spot",
            balances=balances,
            positions={},
            prices={BTC_USDT: 60_000.0},
        )

        assert snapshot_before.get_equity("USDT") == 10_000.0
        assert snapshot_after.get_equity("USDT") == 11_000.0

    def test_equity_includes_locked_quote_balance(self):
        # Locked USDT (e.g. reserved for a pending buy order) counts toward equity
        balances = {"USDT": Balance(locked=3_000.0, total=10_000.0)}
        snapshot = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="spot",
            balances=balances,
            positions={},
            prices={},
        )

        assert snapshot.get_equity("USDT") == 10_000.0

    def test_equity_includes_locked_base_asset(self):
        # Locked BTC (e.g. reserved for a pending sell order) counts toward equity
        balances = {
            "USDT": Balance(locked=0.0, total=5_000.0),
            "BTC": Balance(locked=0.05, total=0.1),  # 0.05 BTC locked
        }
        snapshot = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="spot",
            balances=balances,
            positions={},
            prices={BTC_USDT: 50_000.0},
        )

        # equity = 5000 USDT + 0.1 BTC (total, not just available) * 50000
        assert snapshot.get_equity("USDT") == 10_000.0

    def test_equity_missing_price(self):
        balances = {
            "USDT": Balance(locked=0.0, total=5_000.0),
            "BTC": Balance(locked=0.0, total=0.1),
        }
        snapshot = EquitySnapshot(
            timestamp=TIMESTAMP,
            market_type="spot",
            balances=balances,
            positions={},
            prices={ETH_USDT: 3_000.0},  # BTC price missing
        )

        with pytest.raises(MissingConversionRateError):
            snapshot.get_equity("USDT")
