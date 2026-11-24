import pytest

from robottraderslab import Symbol
from robottraderslab.strategies import profile_tag
from robottraderslab.strategies.futures import TotalBalanceRatio
from robottraderslab_impulse.position_tracking import (
    tags_by_profile_id,
    tracked_symbols,
)
from robottraderslab_impulse.profile_config import ProfileConfig


@pytest.fixture
def btc() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def profile(btc: Symbol) -> ProfileConfig:
    return ProfileConfig(
        sizing=TotalBalanceRatio(1.0),
        symbol=str(btc),
        timeframe="1d",
        trix_length=9,
        signal_length=21,
        trend_length=200,
    )


def _stacked_profiles(btc) -> list[ProfileConfig]:
    return [
        ProfileConfig(
            sizing=TotalBalanceRatio(1.0),
            symbol=str(btc),
            timeframe="1d",
            tag=tag,
            trix_length=9,
            signal_length=21,
            trend_length=200,
        )
        for tag in ("alpha", "beta")
    ]


class TestTrackedSymbols:
    def test_maps_each_profile_to_the_symbol_it_trades(self, profile, btc):
        assert tracked_symbols([profile]) == {profile.profile_id: btc}

    def test_profiles_stacked_on_one_symbol_all_map_to_it(self, btc):
        alpha, beta = _stacked_profiles(btc)

        assert tracked_symbols([alpha, beta]) == {
            alpha.profile_id: btc,
            beta.profile_id: btc,
        }

    def test_no_profiles_map_to_nothing(self):
        assert tracked_symbols([]) == {}


class TestTagsByProfileId:
    def test_maps_each_profile_to_the_tag_its_orders_carry(self, btc):
        alpha, beta = _stacked_profiles(btc)

        assert tags_by_profile_id([alpha, beta]) == {
            alpha.profile_id: profile_tag("1d", "alpha"),
            beta.profile_id: profile_tag("1d", "beta"),
        }
