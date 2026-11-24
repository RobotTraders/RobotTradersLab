from collections.abc import Callable
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import OrderProtocol, OrderType
from robottraderslab.futures import (
    FuturesAccount,
    FuturesCapabilities,
    UpdatePositionTrailingStopAction,
)
from robottraderslab_core.strategies import AccountSnapshot

BTCUSDT = Symbol.create("BTC/USDT:USDT")
TRAIL_SHARE = 0.1


@pytest.fixture
def make_account(make_venue: Callable[..., Mock]) -> Callable[..., FuturesAccount]:
    def make(**capabilities: object) -> FuturesAccount:
        venue = make_venue()
        venue.capabilities = FuturesCapabilities(**capabilities)
        return FuturesAccount(venue)

    return make


@pytest.fixture
def make_snapshot(
    create_resting_order: Callable[..., OrderProtocol],
) -> Callable[..., AccountSnapshot]:
    def make(*resting: tuple[OrderType, float | None]) -> AccountSnapshot:
        return AccountSnapshot(
            account_name="test",
            open_orders=[
                create_resting_order(kind=kind, trigger_price=trigger_price)
                for kind, trigger_price in resting
            ],
        )

    return make


@pytest.mark.parametrize(
    "capabilities",
    [
        {"places_trailing_stop_orders": False},
        {"places_trailing_stop_orders": True},
        {
            "places_trailing_stop_orders": True,
            "min_trailing_share": 0.2,
            "max_trailing_share": 0.3,
        },
    ],
    ids=["offers-none", "offers-one", "share-outside-the-band"],
)
def test_the_booking_is_the_trailing_stop_whatever_the_venue_offers(
    make_account, make_snapshot, capabilities
):
    account = make_account(**capabilities)

    action = account.trail_stop_loss(
        BTCUSDT, TRAIL_SHARE, account_snapshot=make_snapshot()
    )

    assert isinstance(action, UpdatePositionTrailingStopAction)
    assert action.symbol == BTCUSDT
    assert action.distance_ratio == TRAIL_SHARE
    assert action.resting == ()


def test_the_trailing_stops_resting_are_the_ones_carried(make_account, make_snapshot):
    account = make_account(places_trailing_stop_orders=True)

    action = account.trail_stop_loss(
        BTCUSDT,
        TRAIL_SHARE,
        account_snapshot=make_snapshot(
            (OrderType.STOP_LOSS, 90.0), (OrderType.TRAILING_STOP, None)
        ),
    )

    assert [order.kind for order in action.resting] == [OrderType.TRAILING_STOP]


@pytest.mark.parametrize("distance_ratio", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_a_distance_ratio_outside_zero_and_one(
    make_account, make_snapshot, distance_ratio
):
    account = make_account(places_trailing_stop_orders=True)

    with pytest.raises(ValueError, match="`distance_ratio` must be within"):
        account.trail_stop_loss(
            BTCUSDT, distance_ratio, account_snapshot=make_snapshot()
        )
