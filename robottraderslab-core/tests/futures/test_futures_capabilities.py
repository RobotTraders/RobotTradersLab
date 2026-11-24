import pytest

from robottraderslab_core.futures import FuturesCapabilities


def test_default_capabilities_label_trigger_fills():
    assert FuturesCapabilities().labels_trigger_fills


def test_default_capabilities_label_resting_triggers():
    assert FuturesCapabilities().labels_resting_triggers


@pytest.mark.parametrize(
    ("share", "held"),
    [(0.0009, False), (0.001, True), (0.02, True), (0.5, True), (0.51, False)],
)
def test_a_trailing_share_is_held_within_the_band_the_venue_declares(share, held):
    capabilities = FuturesCapabilities(
        places_trailing_stop_orders=True,
        min_trailing_share=0.001,
        max_trailing_share=0.5,
    )

    assert capabilities.holds_trailing_share(share) is held


def test_a_venue_offering_no_trailing_stop_holds_no_share():
    assert not FuturesCapabilities().holds_trailing_share(0.02)
