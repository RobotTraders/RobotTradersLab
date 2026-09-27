import pytest

from robottraderslab.exchanges import PositionSide
from robottraderslab.strategies import TrackedPosition


class TestTrackedPosition:
    def test_creates_long_position_with_valid_quantity(self):
        position = TrackedPosition(side=PositionSide.LONG, quantity=10.5)

        assert position.side == PositionSide.LONG
        assert position.quantity == 10.5

    def test_creates_short_position_with_valid_quantity(self):
        position = TrackedPosition(side=PositionSide.SHORT, quantity=5.2)

        assert position.side == PositionSide.SHORT
        assert position.quantity == 5.2

    def test_raises_error_when_quantity_is_zero(self):
        with pytest.raises(ValueError, match="Quantity must be positive, got: 0"):
            TrackedPosition(side=PositionSide.LONG, quantity=0)

    def test_raises_error_when_quantity_is_negative(self):
        with pytest.raises(ValueError, match="Quantity must be positive, got: -5.0"):
            TrackedPosition(side=PositionSide.LONG, quantity=-5.0)

    def test_position_is_frozen(self):
        position = TrackedPosition(side=PositionSide.LONG, quantity=10.0)

        with pytest.raises(AttributeError):
            position.quantity = 20.0
