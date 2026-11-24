import pandas as pd

from robottraderslab.analyser.profile_names import name_profiles

_BTC = "BTC/USDT:USDT"


def test_a_single_declared_profile_names_every_trade_even_with_absent_or_unrelated_tags():
    trades = pd.DataFrame(
        {
            "symbol": [_BTC, _BTC, _BTC],
            "tag": [None, "rung-1", "unrelated"],
        }
    )
    profiles = [{"symbol": _BTC, "timeframe": "1h", "tag": ""}]

    named = name_profiles(trades, profiles)

    assert list(named["profile_name"]) == [f"{_BTC}@1h"] * 3


def test_a_symbol_with_two_profiles_names_each_trade_by_its_tag():
    trades = pd.DataFrame(
        {
            "symbol": [_BTC, _BTC],
            "tag": ["1h-alpha", "2h-alpha"],
        }
    )
    profiles = [
        {"symbol": _BTC, "timeframe": "1h", "tag": "alpha"},
        {"symbol": _BTC, "timeframe": "2h", "tag": "alpha"},
    ]

    named = name_profiles(trades, profiles)

    assert list(named["profile_name"]) == [f"{_BTC}@1h-alpha", f"{_BTC}@2h-alpha"]


def test_a_tag_matching_no_declared_profile_leaves_the_name_unset():
    trades = pd.DataFrame(
        {
            "symbol": [_BTC, _BTC],
            "tag": ["1h-alpha", "3h-alpha"],
        }
    )
    profiles = [
        {"symbol": _BTC, "timeframe": "1h", "tag": "alpha"},
        {"symbol": _BTC, "timeframe": "2h", "tag": "alpha"},
    ]

    named = name_profiles(trades, profiles)

    assert pd.isna(named["profile_name"].iloc[1])


def test_no_profiles_declared_leaves_the_frame_untouched():
    trades = pd.DataFrame({"symbol": [_BTC], "tag": ["alpha"]})

    named = name_profiles(trades, [])

    pd.testing.assert_frame_equal(named, trades)


def test_a_frame_with_no_trades_is_returned_as_it_is():
    trades = pd.DataFrame({"symbol": [], "tag": []})

    named = name_profiles(trades, [{"symbol": _BTC, "timeframe": "1h", "tag": ""}])

    pd.testing.assert_frame_equal(named, trades)
