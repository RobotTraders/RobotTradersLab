import math
from datetime import datetime

import pytest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator.simulated_position import (
    SimulatedPosition,
    _calculate_average_entry_price,
    _calculate_long_liquidation_price,
    _calculate_short_liquidation_price,
)
from robottraderslab.exchanges import PositionProtocol, PositionSide

# Note on tolerances:
# - We use rel=1e-9 (one part in a billion) for standard values.
# - For cases with billions or many decimal places, we use rel=1e-6
#   to accommodate larger rounding differences in extreme value ranges.

HALF_UNIT_IN_STEPS = 50_000_000
ONE_UNIT_IN_STEPS = 100_000_000
ONE_AND_A_HALF_UNITS_IN_STEPS = 150_000_000
TWO_UNITS_IN_STEPS = 200_000_000


@pytest.fixture
def position() -> SimulatedPosition:
    return SimulatedPosition(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        leverage=1.0,
        taker_fee_rate=0.001,
        entry_time=datetime.fromisoformat("2024-01-01 10:00:00"),
    )


@pytest.fixture
def short_position() -> SimulatedPosition:
    return SimulatedPosition(
        symbol="BTC/USDT",
        side="short",
        leverage=1.0,
        taker_fee_rate=0.001,
        entry_time=datetime.fromisoformat("2024-01-01 10:00:00"),
    )


class TestPositionManagement:
    """Tests for basic position management without focusing on fees."""

    def test_initial_position_setup(self, position):
        """Test basic position creation and properties."""
        assert position.symbol == "BTC/USDT"
        assert position.side == PositionSide.LONG
        assert position.quantity_steps == 0
        assert position.leverage == 1.0
        assert position.taker_fee_rate == 0.001
        assert position.average_entry_price == 0.0
        assert position.liquidation_price == 0.0
        assert position.group_id is None

    @pytest.mark.parametrize(
        ("leverage", "expected_liquidation_price"),
        [
            (1.0, 0.0),
            (5.0, 40000.0),
        ],
    )
    def test_liquidation_price_calculation(
        self, position, leverage, expected_liquidation_price
    ):
        """Test liquidation price calculation with different leverage values."""
        position.leverage = leverage

        net_steps = ONE_UNIT_IN_STEPS
        price = 50000.0
        position.add_to_position(net_steps, price)

        assert position.liquidation_price == pytest.approx(
            expected_liquidation_price, rel=1e-9
        )


class TestCalculateAverageEntryPrice:
    """Tests for average entry price calculation."""

    def test_basic_case_1(self):
        result = _calculate_average_entry_price(1.5, 10, 0.5, 12)
        assert result == pytest.approx(10.5, rel=1e-9)

    def test_basic_case_2(self):
        result = _calculate_average_entry_price(1.5, 10.5, 0.5, 12.5)
        assert result == pytest.approx(11.0, rel=1e-9)

    def test_with_billions(self):
        result = _calculate_average_entry_price(10000.0, 100000, 5000.0, 120000)
        assert result == pytest.approx(106666.67, rel=1e-6)

    def test_with_small_decimals(self):
        result = _calculate_average_entry_price(0.001, 0.000010, 0.0005, 0.000012)
        assert result == pytest.approx(1.0666666666666667e-05, rel=1e-9)

    def test_with_zero_additional_amount(self):
        result = _calculate_average_entry_price(1.0, 10, 0, 12)
        assert result == 10.0

    def test_with_zero_current_amount(self):
        result = _calculate_average_entry_price(0, 10, 1.0, 12)
        assert result == 12.0

    def test_with_nan_price(self):
        result = _calculate_average_entry_price(100, float("nan"), 50, 12)
        assert math.isnan(result)

    def test_with_infinity_price(self):
        result = _calculate_average_entry_price(100, float("inf"), 50, 12)
        assert math.isinf(result)


class TestLiquidationPriceCalculations:
    """Tests for long and short liquidation price calculations."""

    def test_long_standard_case(self):
        assert _calculate_long_liquidation_price(100, 2) == pytest.approx(
            50.0, rel=1e-9
        )

    def test_long_with_realised_pnl(self):
        assert _calculate_long_liquidation_price(
            100_000, 5, realised_pnl=1_000_000
        ) == pytest.approx(-170_000, rel=1e-5)

    def test_long_spot_trading(self):
        assert _calculate_long_liquidation_price(100, 1.0) == 0.0
        assert _calculate_long_liquidation_price(100, 1.0, realised_pnl=50) == 0.0

    def test_short_standard_case(self):
        assert _calculate_short_liquidation_price(100, 2) == pytest.approx(
            150.0, rel=1e-9
        )

    def test_short_with_realised_pnl(self):
        assert _calculate_short_liquidation_price(
            100_000, 5, realised_pnl=1_000_000
        ) == pytest.approx(370_000, rel=1e-5)

    def test_short_spot_trading(self):
        assert _calculate_short_liquidation_price(100, 1.0) == 200.0
        assert _calculate_short_liquidation_price(100, 1.0, realised_pnl=50) == 200.0

    def test_extreme_leverage_cases(self):
        # Long positions
        assert _calculate_long_liquidation_price(100, 100) == pytest.approx(
            99.0, rel=1e-9
        )
        assert _calculate_long_liquidation_price(100, 10000) == pytest.approx(
            99.99, rel=1e-4
        )

        # Short positions
        assert _calculate_short_liquidation_price(100, 100) == pytest.approx(
            101.0, rel=1e-9
        )
        assert _calculate_short_liquidation_price(100, 10000) == pytest.approx(
            100.01, rel=1e-4
        )


class TestGetUnrealisedPnL:
    def test_no_fees_no_profit(self):
        """Test position value with no fees and no price change."""
        position = SimulatedPosition(
            Symbol("BTC", "USDT"),
            PositionSide.LONG,
            leverage=1.0,
            taker_fee_rate=0.0,
            entry_time=datetime.fromisoformat("2024-01-01 10:00:00"),
        )
        position.add_to_position(ONE_UNIT_IN_STEPS, 1.0)
        assert position.get_unrealised_PnL(1.0) == 0.0

    def test_with_profit(self):
        """Test position value with profit."""
        position = SimulatedPosition(
            Symbol("BTC", "USDT"),
            PositionSide.LONG,
            leverage=1.0,
            taker_fee_rate=0.0,
            entry_time=datetime.fromisoformat("2024-01-01 10:00:00"),
        )
        position.add_to_position(ONE_UNIT_IN_STEPS, 1.0)
        assert position.get_unrealised_PnL(2.0) == 1.0

    def test_with_loss(self):
        """Test position value with loss."""
        position = SimulatedPosition(
            Symbol("BTC", "USDT"),
            PositionSide.LONG,
            leverage=1.0,
            taker_fee_rate=0.0,
            entry_time=datetime.fromisoformat("2024-01-01 10:00:00"),
        )
        position.add_to_position(ONE_UNIT_IN_STEPS, 2.0)
        assert position.get_unrealised_PnL(1.0) == -1.0

    def test_with_leverage(self):
        """Test position value with leverage."""
        position = SimulatedPosition(
            Symbol("BTC", "USDT"),
            PositionSide.LONG,
            leverage=10.0,
            taker_fee_rate=0.0,
            entry_time=datetime.fromisoformat("2024-01-01 10:00:00"),
        )
        position.add_to_position(ONE_UNIT_IN_STEPS, 100.0)
        assert position.get_unrealised_PnL(110.0) == 10.0

    def test_short_position_with_profit(self):
        """Test short position value with profit."""
        position = SimulatedPosition(
            Symbol("BTC", "USDT"),
            PositionSide.SHORT,
            leverage=1.0,
            taker_fee_rate=0.0,
            entry_time=datetime.fromisoformat("2024-01-01 10:00:00"),
        )
        position.add_to_position(ONE_UNIT_IN_STEPS, 2.0)
        assert position.get_unrealised_PnL(1.0) == 1.0


class TestEntryTime:
    """Tests for entry_time functionality in SimulatedPosition."""

    def test_entry_time_is_set_on_creation(self):
        """Test that entry_time is properly set when creating a position."""
        entry_time = datetime.fromisoformat("2024-01-15 10:30:00")
        position = SimulatedPosition(
            Symbol("BTC", "USDT"),
            PositionSide.LONG,
            leverage=1.0,
            taker_fee_rate=0.001,
            entry_time=entry_time,
        )
        assert position.entry_time == entry_time

    def test_entry_time_unchanged_after_adding_to_position(self):
        """Test that entry_time remains unchanged when adding to position."""
        entry_time = datetime.fromisoformat("2024-01-15 10:30:00")
        position = SimulatedPosition(
            Symbol("BTC", "USDT"),
            PositionSide.LONG,
            leverage=1.0,
            taker_fee_rate=0.001,
            entry_time=entry_time,
        )

        net_steps1, price1 = ONE_UNIT_IN_STEPS, 50000.0
        position.add_to_position(net_steps1, price1)
        assert position.entry_time == entry_time

        net_steps2, price2 = HALF_UNIT_IN_STEPS, 51000.0
        position.add_to_position(net_steps2, price2)
        assert position.entry_time == entry_time

    def test_entry_time_unchanged_after_reducing_position(self):
        """Test that entry_time remains unchanged when reducing position."""
        entry_time = datetime.fromisoformat("2024-01-15 10:30:00")
        position = SimulatedPosition(
            Symbol("BTC", "USDT"),
            PositionSide.LONG,
            leverage=1.0,
            taker_fee_rate=0.001,
            entry_time=entry_time,
        )
        position.add_to_position(TWO_UNITS_IN_STEPS, 50000.0)

        position.reduce_position(HALF_UNIT_IN_STEPS)
        assert position.entry_time == entry_time
        assert position.quantity_steps == ONE_AND_A_HALF_UNITS_IN_STEPS

        position.reduce_position(position.quantity_steps)
        assert position.entry_time == entry_time
        assert position.quantity_steps == 0

    def test_entry_time_accessible_via_protocol(self):
        """Test that entry_time is accessible through PositionProtocol interface."""
        entry_time = datetime.fromisoformat("2024-01-15 10:30:00")
        position: PositionProtocol = SimulatedPosition(
            Symbol("BTC", "USDT"),
            PositionSide.LONG,
            leverage=1.0,
            taker_fee_rate=0.001,
            entry_time=entry_time,
        )

        # Should be accessible through protocol interface
        assert position.entry_time == entry_time
