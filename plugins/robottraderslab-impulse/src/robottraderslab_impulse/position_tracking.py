from collections.abc import Sequence

from robottraderslab import Symbol
from robottraderslab.strategies import ProfileTag, TrackingId

from .profile_config import ProfileConfig


def tags_by_profile_id(
    profiles: Sequence[ProfileConfig],
) -> dict[TrackingId, ProfileTag]:
    """Return the tag each profile's own orders carry."""
    return {profile.profile_id: profile.order_tag for profile in profiles}


def tracked_symbols(profiles: Sequence[ProfileConfig]) -> dict[TrackingId, Symbol]:
    """Return the symbol each profile's tracked position trades."""
    return {profile.profile_id: profile.symbol for profile in profiles}
