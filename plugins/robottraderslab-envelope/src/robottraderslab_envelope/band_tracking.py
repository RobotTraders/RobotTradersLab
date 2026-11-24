from collections.abc import Sequence

from robottraderslab import Symbol
from robottraderslab.strategies import (
    PositionSide,
    PositionTracker,
    ProfileTag,
    TrackingId,
)

from .profile_config import ProfileConfig


def held_bands(
    profile: ProfileConfig, side: PositionSide, tracker: PositionTracker
) -> frozenset[int]:
    return frozenset(
        band_index
        for band_index, tracking_id in enumerate(
            profile.band_tracking_ids[side], start=1
        )
        if tracker.get(tracking_id) is not None
    )


def tags_by_tracking_id(
    profiles: Sequence[ProfileConfig],
) -> dict[TrackingId, ProfileTag]:
    return {
        tracking_id: tag
        for profile in profiles
        for side, tags in profile.rung_tags.items()
        for tracking_id, tag in zip(profile.band_tracking_ids[side], tags, strict=True)
    }


def tracked_symbols(profiles: Sequence[ProfileConfig]) -> dict[TrackingId, Symbol]:
    return {
        tracking_id: profile.symbol
        for profile in profiles
        for tracking_ids in profile.band_tracking_ids.values()
        for tracking_id in tracking_ids
    }
