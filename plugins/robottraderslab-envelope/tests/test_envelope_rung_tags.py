import pytest
from rung_ids import PROCESS_TAG

from robottraderslab.strategies import PositionSide
from robottraderslab_envelope.rungs import rung_band_index


class TestRungBandIndex:
    @pytest.mark.parametrize(
        ("client_order_id", "side", "band_index"),
        [
            (f"{PROCESS_TAG}-1d-l1", PositionSide.LONG, 1),
            (f"{PROCESS_TAG}-1d-l3", PositionSide.LONG, 3),
            ("--1d-s1", PositionSide.SHORT, 1),
        ],
    )
    def test_a_rung_tagged_for_a_band_of_the_profile_on_its_side(
        self, profiles, client_order_id, side, band_index
    ):
        assert rung_band_index(profiles[0], side, client_order_id) == band_index

    @pytest.mark.parametrize(
        ("client_order_id", "side"),
        [
            (f"{PROCESS_TAG}-r1", PositionSide.LONG),
            (f"{PROCESS_TAG}-1d-r1", PositionSide.LONG),
            (f"{PROCESS_TAG}-1d-r1", PositionSide.SHORT),
            (f"{PROCESS_TAG}-1d-s1", PositionSide.LONG),
            (f"{PROCESS_TAG}-1d-l1", PositionSide.SHORT),
            (f"{PROCESS_TAG}-1d-alpha", PositionSide.LONG),
            (f"{PROCESS_TAG}-1d-l4", PositionSide.LONG),
            (f"{PROCESS_TAG}-1d-s2", PositionSide.SHORT),
            (f"{PROCESS_TAG}-1d-lx", PositionSide.LONG),
            (f"{PROCESS_TAG}-4h-l2", PositionSide.LONG),
            (f"{PROCESS_TAG}-1d", PositionSide.LONG),
            ("t1785908301620", PositionSide.LONG),
            (None, PositionSide.LONG),
        ],
        ids=[
            "bare-tag",
            "tag-shared-by-both-sides-long",
            "tag-shared-by-both-sides-short",
            "short-tag-on-the-long-side",
            "long-tag-on-the-short-side",
            "other-tag",
            "long-band-beyond-the-ladder",
            "short-band-beyond-the-ladder",
            "no-digits",
            "other-timeframe",
            "no-custom",
            "foreign",
            "none",
        ],
    )
    def test_an_id_carrying_no_tag_of_the_profiles_rungs_on_that_side(
        self, profiles, client_order_id, side
    ):
        assert rung_band_index(profiles[0], side, client_order_id) is None


class TestProfilesOnDifferentTimeframes:
    def test_each_profile_claims_only_the_rungs_under_its_own_tags(self, profiles):
        daily, hourly = profiles
        long = PositionSide.LONG

        assert rung_band_index(daily, long, f"{PROCESS_TAG}-1d-l2") == 2
        assert rung_band_index(hourly, long, f"{PROCESS_TAG}-1h-l2") == 2
        assert rung_band_index(daily, long, f"{PROCESS_TAG}-1h-l2") is None
        assert rung_band_index(hourly, long, f"{PROCESS_TAG}-1d-l2") is None
