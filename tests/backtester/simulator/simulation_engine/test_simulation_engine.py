from datetime import datetime

import pytest

from robottraderslab import Symbol
from robottraderslab._core import OHLCVRow
from robottraderslab.backtester.simulator import FeeRates
from robottraderslab.exchanges import Balance, OrderSide


def _ohlcv(close: float) -> OHLCVRow:
    return OHLCVRow(
        timestamp=datetime.now(),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=0.0,
    )


class TestInitialization:
    @pytest.fixture
    def fee_rates(self) -> FeeRates:
        return FeeRates(maker=10.0, taker=20.0)

    def test_init(self, sim, fee_rates: FeeRates):
        assert sim.simulation_engine._fee_rates == fee_rates
        assert sim.simulation_engine._last_seen_prices == {}
        assert sim.simulation_engine._last_tick_snapshot is None
        assert sim.simulation_engine._leverage_by_symbol == {}
        assert sim.simulation_engine._market_type == "futures"
        assert sim.simulation_engine._trades_occurred_this_tick is False

        assert sim.simulation_engine.margin_currency == "USDT"
        assert sim.simulation_engine.open_orders == []
        assert sim.simulation_engine.open_positions == {}

        assert sim.simulation_engine.get_balances() == {
            "USDT": Balance(locked=0.0, total=sim.initial_usdt)
        }
        assert sim.simulation_engine.get_equity("USDT") == sim.initial_usdt
        assert sim.simulation_engine.get_daily_equity_snapshots() == []
        assert sim.simulation_engine.get_trade_equity_snapshots() == []
        assert sim.simulation_engine.get_last_fills() == {}


class TestLastTickSnapshot:
    def test_set_by_simulation(self, sim):
        timestamp = datetime.fromisoformat("2024-02-16 12:30:00")
        sim.simulation_engine.simulate_on_current_ohlcvs(timestamp, {})

        last_tick_snapshot = sim.simulation_engine._last_tick_snapshot
        assert last_tick_snapshot.timestamp == timestamp
        assert last_tick_snapshot.market_type == "futures"
        assert last_tick_snapshot.balances["USDT"].total == 10_000.0
        assert last_tick_snapshot.positions == {}

    def test_uses_last_seen_prices(
        self, sim, btc_usdt_perp: Symbol, eth_usdt_perp: Symbol
    ):
        ohlcvs_by_symbol = {btc_usdt_perp: _ohlcv(100.0)}
        sim.simulation_engine.simulate_on_current_ohlcvs(
            datetime.fromisoformat("2024-02-16 12:30:00"), ohlcvs_by_symbol
        )
        assert sim.simulation_engine._last_tick_snapshot.prices == {
            btc_usdt_perp: 100.0
        }

        ohlcvs_by_symbol = {
            btc_usdt_perp: _ohlcv(50.0),
            eth_usdt_perp: _ohlcv(150.0),
        }
        sim.simulation_engine.simulate_on_current_ohlcvs(
            datetime.fromisoformat("2024-02-16 13:30:00"), ohlcvs_by_symbol
        )
        assert sim.simulation_engine._last_tick_snapshot.prices == {
            btc_usdt_perp: 50.0,
            eth_usdt_perp: 150.0,
        }


class TestLastSeenPrices:
    def test_acquires_prices(self, sim, btc_usdt_perp: Symbol):
        timestamp = datetime.fromisoformat("2024-02-16 12:30:00")
        ohlcvs_by_symbol = {btc_usdt_perp: _ohlcv(100.0)}

        sim.simulation_engine.simulate_on_current_ohlcvs(timestamp, ohlcvs_by_symbol)

        assert sim.simulation_engine._last_seen_prices == {btc_usdt_perp: 100.0}

    def test_updates_prices(self, sim, btc_usdt_perp: Symbol, eth_usdt_perp: Symbol):
        timestamp = datetime.fromisoformat("2024-02-16 12:30:00")
        ohlcvs_by_symbol = {btc_usdt_perp: _ohlcv(100.0)}
        sim.simulation_engine.simulate_on_current_ohlcvs(timestamp, ohlcvs_by_symbol)

        ohlcvs_by_symbol = {
            btc_usdt_perp: _ohlcv(50.0),
            eth_usdt_perp: _ohlcv(150.0),
        }
        sim.simulation_engine.simulate_on_current_ohlcvs(timestamp, ohlcvs_by_symbol)

        assert sim.simulation_engine._last_seen_prices == {
            btc_usdt_perp: 50.0,
            eth_usdt_perp: 150.0,
        }


class TestLastVenueFill:
    async def test_when_market_order_is_executed(self, sim, btc_usdt_perp: Symbol):
        placed_order = await sim.exchange.place_market_order(
            btc_usdt_perp, OrderSide.BUY, 4.0
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        assert placed_order.order_id in sim.simulation_engine.get_last_fills()

    async def test_executed_order_last_only_one_iteration(
        self, sim, btc_usdt_perp: Symbol
    ):
        placed_order = await sim.exchange.place_market_order(
            btc_usdt_perp, OrderSide.BUY, 4.0
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        assert placed_order.order_id not in sim.simulation_engine.get_last_fills()

    async def test_market_order_provides_executed_filled_quantity(
        self, sim, btc_usdt_perp: Symbol
    ):
        sim.simulation_engine._fee_rates = FeeRates(maker=0.1, taker=0.1)
        placed_order = await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=1.5,
        )

        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        order_fill = await placed_order.get_fill()

        assert order_fill.quantity < 1.5


class TestOpenPositionsReturnsImmutableSnapshots:
    """Regression test for issue #512."""

    async def test_snapshot_is_not_affected_by_position_closure(
        self, sim, btc_usdt_perp: Symbol
    ):
        await sim.exchange.place_market_order(btc_usdt_perp, OrderSide.BUY, 1.0)
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        snapshot = sim.open_positions[btc_usdt_perp]
        original_quantity = snapshot.quantity
        assert original_quantity > 0

        await sim.exchange.place_market_order(
            btc_usdt_perp, OrderSide.SELL, original_quantity, reduce_only=True
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=100.0)

        assert snapshot.quantity == original_quantity
        assert btc_usdt_perp not in sim.open_positions
