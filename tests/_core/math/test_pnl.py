import math

import pytest

from robottraderslab._core import (
    calculate_pnl_long,
    calculate_pnl_short,
    calculate_trade_pnl_pct,
)


def test_calculate_pnl_long():
    """Test PnL calculations for long positions."""
    # Standard long position PnL
    # Formula: (exit_price - entry_price) * quantity

    # Profitable trade
    # For entry=100, exit=120, quantity=1: (120-100)*1 = 20
    assert calculate_pnl_long(100, 120, 1.0) == 20.0

    # Loss trade
    # For entry=100, exit=80, quantity=1: (80-100)*1 = -20
    assert calculate_pnl_long(100, 80, 1.0) == -20.0

    # Large numbers test
    # For entry=50,000, exit=100,000, quantity=100: (100,000-50,000)*100 = 5,000,000
    assert calculate_pnl_long(50_000, 100_000, 100.0) == 5_000_000.0

    # Small numbers test
    # For entry=1e-8, exit=2e-8, quantity=1e6: (2e-8-1e-8)*1e6 = 0.01
    assert calculate_pnl_long(0.00000001, 0.00000002, 1_000_000.0) == pytest.approx(
        0.01, rel=1e-5
    )

    # Negative entry with positive exit (crossing zero)
    assert calculate_pnl_long(-10, 20, 1.0) == 30.0

    # Zero quantity test
    assert calculate_pnl_long(100, 120, 0.0) == 0.0

    # Zero price test
    assert calculate_pnl_long(0, 10, 1.0) == 10.0


def test_calculate_pnl_short():
    """Test PnL calculations for short positions."""
    # Standard short position PnL
    # Formula: (entry_price - exit_price) * quantity

    # Profitable trade
    # For entry=100, exit=80, quantity=1: (100-80)*1 = 20
    assert calculate_pnl_short(100, 80, 1.0) == 20.0

    # Loss trade
    # For entry=100, exit=120, quantity=1: (100-120)*1 = -20
    assert calculate_pnl_short(100, 120, 1.0) == -20.0

    # Large numbers test
    # For entry=100,000, exit=50,000, quantity=100: (100,000-50,000)*100 = 5,000,000
    assert calculate_pnl_short(100_000, 50_000, 100.0) == 5_000_000.0

    # Small numbers test
    # For entry=2e-8, exit=1e-8, quantity=1e6: (2e-8-1e-8)*1e6 = 0.01
    assert calculate_pnl_short(0.00000002, 0.00000001, 1_000_000.0) == pytest.approx(
        0.01, rel=1e-5
    )

    # Negative entry with positive exit (crossing zero)
    assert calculate_pnl_short(-10, 20, 1.0) == -30.0

    # Zero quantity test
    assert calculate_pnl_short(100, 80, 0.0) == 0.0

    # Zero price test
    assert calculate_pnl_short(0, 10, 1.0) == -10.0


def test_calculate_trade_pnl_pct():
    """Test percentage PnL calculations."""
    # Standard PnL percentage calculation
    # Formula: (PnL / (entry_price * quantity)) * 100

    # Profitable trade
    # For PnL=10, entry=100, quantity=1: (10/(100*1)) = 0.10 (10%)
    assert calculate_trade_pnl_pct(10, 100, 1.0) == 0.10

    # Loss trade
    # For PnL=-10, entry=100, quantity=1: (-10/(100*1)) = -0.10 (-10%)
    assert calculate_trade_pnl_pct(-10, 100, 1.0) == -0.10

    # Large numbers test
    # For PnL=1,000,000, entry=100,000, quantity=100: (1,000,000/(100,000*100)) = 0.10 (10%)
    assert calculate_trade_pnl_pct(1_000_000, 100_000, 100.0) == 0.10

    # Small numbers test
    # For PnL=1e-8, entry=1e-7, quantity=1e6: (1e-8/(1e-7*1e6)) = 0.0000001 (0.00001%)
    assert calculate_trade_pnl_pct(0.00000001, 0.0000001, 1_000_000.0) == pytest.approx(
        0.0000001, rel=1e-5
    )

    # Test with zero entry price (should raise ZeroDivisionError)
    with pytest.raises(ZeroDivisionError):
        calculate_trade_pnl_pct(10, 0, 1.0)


def test_large_value_performance():
    """Test that calculations with large values are handled efficiently."""
    # Test with extremely large amounts that could potentially cause overflow
    large_amount = 1e15
    large_price = 1e15
    result = calculate_pnl_long(large_price, large_price * 1.01, large_amount)
    assert not math.isnan(result)
    assert not math.isinf(result)

    # Test with extremely small amounts that could cause underflow
    tiny_amount = 1e-15
    tiny_price = 1e-15
    result = calculate_pnl_long(tiny_price, tiny_price * 1.01, tiny_amount)
    assert not math.isnan(result)
    assert not math.isinf(result)

    # Test very large differences between entry and exit prices
    result = calculate_pnl_long(1, 1000000, 1)
    assert result == 999999  # Should handle large disparities correctly


@pytest.mark.parametrize(
    "test_params",
    [
        # Normal case - Long position
        {
            "quantity": 10.0,
            "close_price": 100.0,
            "entry_price": 90.0,
            "expected_pnl": 100.0,  # (100-90)*10
            "expected_pnl_pct": 0.11111111,  # (100/900)
        },
        # Edge case - Small numbers
        {
            "quantity": 0.00001,
            "close_price": 0.00001,
            "entry_price": 0.000009,
            "expected_pnl": 1e-11,  # (0.00001-0.000009)*0.00001 = (1e-5 - 9e-6)*1e-5
            "expected_pnl_pct": 0.11111111,  # Same percentage as normal case
        },
        # Edge case - Large numbers
        {
            "quantity": 1000000.0,
            "close_price": 100000.0,
            "entry_price": 90000.0,
            "expected_pnl": 10000000000.0,  # (100000-90000)*1000000
            "expected_pnl_pct": 0.11111111,  # Same percentage as normal case
        },
    ],
)
def test_pnl_calculation_scenarios(test_params):
    """Test PnL calculations with various scenarios and number scales."""
    # Calculate actual PnL
    actual_pnl = calculate_pnl_long(
        test_params["entry_price"],
        test_params["close_price"],
        test_params["quantity"],
    )

    # Calculate actual PnL percentage
    actual_pnl_pct = calculate_trade_pnl_pct(
        actual_pnl,
        test_params["entry_price"],
        test_params["quantity"],
    )

    # Verify results
    assert actual_pnl == pytest.approx(test_params["expected_pnl"], rel=1e-6)
    assert actual_pnl_pct == pytest.approx(test_params["expected_pnl_pct"], rel=1e-6)
