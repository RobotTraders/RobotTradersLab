from robottraderslab.exchanges import FuturesExchangeProtocol


class _MarginOnlyVenue(FuturesExchangeProtocol):
    pass


def test_a_venue_reserving_nothing_locks_the_margin_alone():
    assert _MarginOnlyVenue().placement_requirement_rate(4.0) == 0.25
