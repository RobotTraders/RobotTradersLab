from dataclasses import FrozenInstanceError
from datetime import datetime

import pandas as pd
import pytest

from robottraderslab.analyser.trade_filter import TradeFilter


@pytest.fixture
def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [
                "BTC/USDT:USDT",
                "BTC/USDT:USDT",
                "ETH/USDT:USDT",
                "ETH/USDT:USDT",
                "BTC/USDT:USDT",
                "ETH/USDT:USDT",
            ],
            "side": ["long", "short", "long", "short", "long", "long"],
            "entry_time": pd.to_datetime(
                [
                    datetime(2024, 1, 1, 10, 0),
                    datetime(2024, 1, 1, 11, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 13, 0),
                    datetime(2024, 1, 1, 14, 0),
                    datetime(2024, 1, 1, 15, 0),
                ]
            ),
            "net_pnl": [100.0, -50.0, 75.0, -25.0, 150.0, 80.0],
        }
    )


@pytest.fixture
def trades_with_profile_name() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [
                "BTC/USDT:USDT",
                "BTC/USDT:USDT",
                "ETH/USDT:USDT",
                "ETH/USDT:USDT",
                "BTC/USDT:USDT",
                "ETH/USDT:USDT",
            ],
            "side": ["long", "short", "long", "short", "long", "long"],
            "profile_name": [
                "BTC/USDT:USDT@5m_aggressive",
                "BTC/USDT:USDT@1h_conservative",
                "ETH/USDT:USDT@5m_aggressive",
                "ETH/USDT:USDT@15m_neutral",
                "BTC/USDT:USDT@1h_aggressive",
                None,
            ],
            "net_pnl": [100.0, -50.0, 75.0, -25.0, 150.0, 80.0],
        }
    )


def test_filter_is_frozen():
    trade_filter = TradeFilter(side="long")

    with pytest.raises(FrozenInstanceError):
        trade_filter.side = "short"


def test_apply_filter_with_no_criteria(trades):
    trade_filter = TradeFilter()

    kept = trade_filter.apply_filter(trades)

    assert len(kept) == len(trades)
    pd.testing.assert_frame_equal(kept, trades)


@pytest.mark.parametrize(
    ("side", "expected_count"),
    [
        ("long", 4),
        ("short", 2),
    ],
)
def test_apply_filter_by_side(trades, side, expected_count):
    trade_filter = TradeFilter(side=side)

    kept = trade_filter.apply_filter(trades)

    assert len(kept) == expected_count
    assert all(kept["side"] == side)


def test_apply_filter_by_symbol(trades):
    trade_filter = TradeFilter(symbol="BTC/USDT:USDT")

    kept = trade_filter.apply_filter(trades)

    assert len(kept) == 3
    assert all(kept["symbol"] == "BTC/USDT:USDT")


def test_apply_filter_by_side_and_symbol(trades):
    trade_filter = TradeFilter(side="long", symbol="ETH/USDT:USDT")

    kept = trade_filter.apply_filter(trades)

    assert len(kept) == 2
    assert all(kept["side"] == "long")
    assert all(kept["symbol"] == "ETH/USDT:USDT")


def test_apply_filter_returns_copy(trades):
    trade_filter = TradeFilter(side="long")

    kept = trade_filter.apply_filter(trades)
    kept.loc[kept.index[0], "net_pnl"] = 999.0

    assert trades.loc[trades.index[0], "net_pnl"] != 999.0


def test_apply_filter_on_empty_dataframe():
    empty_df = pd.DataFrame(columns=["symbol", "side", "net_pnl"])
    trade_filter = TradeFilter(side="long")

    kept = trade_filter.apply_filter(empty_df)

    assert len(kept) == 0
    assert list(kept.columns) == list(empty_df.columns)


def test_apply_filter_with_missing_side_column():
    df_without_side = pd.DataFrame({"symbol": ["BTC/USDT:USDT"], "net_pnl": [100.0]})
    trade_filter = TradeFilter(side="long")

    with pytest.raises(ValueError, match="Required columns.*side.*not found"):
        trade_filter.apply_filter(df_without_side)


def test_apply_filter_with_missing_symbol_column():
    df_without_symbol = pd.DataFrame({"side": ["long"], "net_pnl": [100.0]})
    trade_filter = TradeFilter(symbol="BTC/USDT:USDT")

    with pytest.raises(ValueError, match="Required columns.*symbol.*not found"):
        trade_filter.apply_filter(df_without_symbol)


def test_apply_filter_with_missing_profile_name_column():
    df_without_profile_name = pd.DataFrame(
        {"symbol": ["BTC/USDT:USDT"], "side": ["long"], "net_pnl": [100.0]}
    )
    trade_filter = TradeFilter(profile_name_contains="test")

    with pytest.raises(ValueError, match="Required columns.*profile_name.*not found"):
        trade_filter.apply_filter(df_without_profile_name)


def test_apply_filter_with_no_matching_trades(trades):
    trade_filter = TradeFilter(symbol="NONEXISTENT/USDT:USDT")

    kept = trade_filter.apply_filter(trades)

    assert len(kept) == 0
    assert list(kept.columns) == list(trades.columns)


def test_apply_filter_preserves_index(trades):
    trade_filter = TradeFilter(side="long")

    kept = trade_filter.apply_filter(trades)
    long_indices = trades[trades["side"] == "long"].index

    assert all(kept.index == long_indices)


def test_apply_filter_by_profile_name_single_pattern(trades_with_profile_name):
    trade_filter = TradeFilter(profile_name_contains="aggressive")

    kept = trade_filter.apply_filter(trades_with_profile_name)

    assert len(kept) == 3
    assert all("aggressive" in name for name in kept["profile_name"])


def test_apply_filter_by_profile_name_multiple_patterns_uses_and_logic(
    trades_with_profile_name,
):
    trade_filter = TradeFilter(profile_name_contains=["BTC", "aggressive"])

    kept = trade_filter.apply_filter(trades_with_profile_name)

    assert len(kept) == 2
    assert all("BTC" in name and "aggressive" in name for name in kept["profile_name"])


def test_apply_filter_by_profile_name_and_logic_partial_match_returns_empty(
    trades_with_profile_name,
):
    trade_filter = TradeFilter(profile_name_contains=["BTC", "nonexistent"])

    kept = trade_filter.apply_filter(trades_with_profile_name)

    assert len(kept) == 0


def test_apply_filter_by_profile_name_or_logic_via_regex(trades_with_profile_name):
    trade_filter = TradeFilter(profile_name_contains="5m|15m")

    kept = trade_filter.apply_filter(trades_with_profile_name)

    assert len(kept) == 3
    assert all("5m" in name or "15m" in name for name in kept["profile_name"])


def test_apply_filter_by_profile_name_excludes_none(trades_with_profile_name):
    trade_filter = TradeFilter(profile_name_contains="ETH")

    kept = trade_filter.apply_filter(trades_with_profile_name)

    assert len(kept) == 2
    assert all(name is not None for name in kept["profile_name"])


def test_apply_filter_by_profile_name_no_matches(trades_with_profile_name):
    trade_filter = TradeFilter(profile_name_contains="nonexistent")

    kept = trade_filter.apply_filter(trades_with_profile_name)

    assert len(kept) == 0


def test_apply_filter_by_profile_name_case_sensitive(trades_with_profile_name):
    trade_filter = TradeFilter(profile_name_contains="AGGRESSIVE")

    kept = trade_filter.apply_filter(trades_with_profile_name)

    assert len(kept) == 0


def test_apply_filter_combined_side_and_profile_name(trades_with_profile_name):
    trade_filter = TradeFilter(side="long", profile_name_contains="aggressive")

    kept = trade_filter.apply_filter(trades_with_profile_name)

    assert len(kept) == 3
    assert all(kept["side"] == "long")
    assert all("aggressive" in name for name in kept["profile_name"])


def test_apply_filter_combined_symbol_and_profile_name(trades_with_profile_name):
    trade_filter = TradeFilter(symbol="BTC/USDT:USDT", profile_name_contains="1h")

    kept = trade_filter.apply_filter(trades_with_profile_name)

    assert len(kept) == 2
    assert all(kept["symbol"] == "BTC/USDT:USDT")
    assert all("1h" in name for name in kept["profile_name"])


def test_apply_filter_combined_all_criteria(trades_with_profile_name):
    trade_filter = TradeFilter(
        side="long", symbol="BTC/USDT:USDT", profile_name_contains="aggressive"
    )

    kept = trade_filter.apply_filter(trades_with_profile_name)

    assert len(kept) == 2
    assert all(kept["side"] == "long")
    assert all(kept["symbol"] == "BTC/USDT:USDT")
    assert all("aggressive" in name for name in kept["profile_name"])


def test_is_active_with_no_filters():
    trade_filter = TradeFilter()

    assert trade_filter.is_active is False


@pytest.mark.parametrize(
    "criteria",
    [
        {"side": "long"},
        {"symbol": "BTC/USDT:USDT"},
        {"side": "short", "symbol": "ETH/USDT:USDT"},
        {"profile_name_contains": "aggressive"},
        {"profile_name_contains": ["5m", "1h"]},
    ],
)
def test_is_active_with_any_criterion(criteria):
    assert TradeFilter(**criteria).is_active is True


def test_description_with_no_filters():
    trade_filter = TradeFilter()

    assert trade_filter.description == "no_filter"


@pytest.mark.parametrize(
    ("side", "expected_desc"),
    [
        ("long", "side_long"),
        ("short", "side_short"),
    ],
)
def test_description_with_side_filter(side, expected_desc):
    trade_filter = TradeFilter(side=side)

    assert trade_filter.description == expected_desc


def test_description_with_symbol_filter():
    trade_filter = TradeFilter(symbol="BTC/USDT:USDT")

    assert trade_filter.description == "symbol_BTC_USDT_USDT"


def test_description_with_combined_filters():
    trade_filter = TradeFilter(side="long", symbol="ETH/USDT:USDT")

    assert trade_filter.description == "side_long_symbol_ETH_USDT_USDT"


def test_description_with_profile_name_single_pattern():
    trade_filter = TradeFilter(profile_name_contains="aggressive")

    assert trade_filter.description == "profile_aggressive"


def test_description_with_profile_name_multiple_patterns():
    trade_filter = TradeFilter(profile_name_contains=["5m", "15m"])

    assert trade_filter.description == "profile_5m_and_15m"


def test_description_with_all_filters():
    trade_filter = TradeFilter(
        side="long", symbol="BTC/USDT:USDT", profile_name_contains="aggressive"
    )

    assert (
        trade_filter.description == "side_long_symbol_BTC_USDT_USDT_profile_aggressive"
    )
