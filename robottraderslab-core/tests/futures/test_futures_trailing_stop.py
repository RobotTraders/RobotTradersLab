import pytest

from robottraderslab_core import Symbol
from robottraderslab_core.exceptions import BotCriticalError
from robottraderslab_core.futures import FuturesCapabilities
from robottraderslab_core.futures.futures_trailing_stop import trailing_percent

BTCUSDT = Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def capabilities() -> FuturesCapabilities:
    return FuturesCapabilities(
        places_trailing_stop_orders=True,
        min_trailing_share=0.001,
        max_trailing_share=0.5,
    )


@pytest.mark.parametrize(
    ("share", "percent"),
    [
        (0.02, "2.00"),
        (0.0125, "1.25"),
        (0.01236, "1.24"),
        (0.001, "0.10"),
        (0.5, "50.00"),
    ],
)
def test_a_share_is_a_percentage_at_two_decimals(capabilities, share, percent):
    assert trailing_percent(capabilities, BTCUSDT, share) == percent


@pytest.mark.parametrize("share", [0.0, 0.0009, 0.51, 1.0])
def test_a_share_outside_the_band_the_venue_declares(capabilities, share):
    with pytest.raises(BotCriticalError, match="outside the 0.1% to 50%"):
        trailing_percent(capabilities, BTCUSDT, share)
