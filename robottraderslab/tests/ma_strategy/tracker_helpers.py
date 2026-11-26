from unittest.mock import Mock

from robottraderslab.strategies import PositionTracker, TrackedPosition


def tracker_reporting(position: TrackedPosition | None) -> Mock:
    """Return a position tracker that reports `position` for every profile.

    Args:
        position: The tracked position it reports; None reports no position.
    """
    position_tracker = Mock(spec=PositionTracker)
    position_tracker.get.return_value = position
    return position_tracker
