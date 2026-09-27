from datetime import datetime

import pandas as pd
import pytest

from robottraderslab.analyser.trade_record import MatchedAmounts, TradeRecord


@pytest.fixture
def entry_fill() -> pd.Series:
    return pd.Series(
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "timestamp": datetime(2024, 1, 1, 12, 0),
            "price": 45000.0,
            "gross_quantity": 1.0,
            "net_quantity": 0.999,
            "fee": 0.001,
        }
    )


@pytest.fixture
def exit_fill() -> pd.Series:
    return pd.Series(
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "timestamp": datetime(2024, 1, 2, 12, 0),
            "price": 46000.0,
            "gross_quantity": 1.0,
            "net_quantity": 0.999,
            "fee": 0.001,
        }
    )


@pytest.fixture
def matched_amounts() -> MatchedAmounts:
    return MatchedAmounts(net=0.4995, gross=0.5, entry_fee=0.0005, exit_fee=0.0005)


@pytest.fixture
def trade_record(entry_fill) -> TradeRecord:
    return TradeRecord.from_entry(entry_fill)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("symbol", "BTC/USDT"),
        ("side", "long"),
        ("entry_time", datetime(2024, 1, 1, 12, 0)),
        ("entry_price", 45000.0),
        ("gross_quantity", 1.0),
        ("net_quantity", 0.999),
        ("entry_fee", 0.001),
    ],
)
def test_from_entry_sets_field_correctly(entry_fill, field, expected):
    trade = TradeRecord.from_entry(entry_fill)
    assert getattr(trade, field) == expected


def test_from_entry_initialises_exit_fields(entry_fill):
    trade = TradeRecord.from_entry(entry_fill)
    assert trade.exit_time is None
    assert trade.exit_price == 0.0
    assert trade.exit_fee == 0.0
    assert trade.gross_pnl == 0.0
    assert trade.net_pnl == 0.0
    assert trade.net_pnl_pct == 0.0


@pytest.mark.parametrize(
    ("field", "key"),
    [
        ("exit_time", "timestamp"),
        ("exit_price", "price"),
        ("gross_quantity", "gross"),
        ("net_quantity", "net"),
        ("entry_fee", "entry_fee"),
        ("exit_fee", "exit_fee"),
    ],
)
def test_fill_exit_updates_fields(trade_record, exit_fill, matched_amounts, field, key):
    trade_record.fill_exit(exit_fill, matched_amounts)

    expected = (
        exit_fill[key]
        if key in ("timestamp", "price")
        else getattr(matched_amounts, key)
    )
    assert getattr(trade_record, field) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", "BTC/USDT"),
        ("side", "long"),
        ("entry_time", datetime(2024, 1, 1, 12, 0)),
        ("entry_price", 45000.0),
    ],
)
def test_fill_exit_preserves_entry_fields(
    trade_record, exit_fill, matched_amounts, field, value
):
    trade_record.fill_exit(exit_fill, matched_amounts)
    assert getattr(trade_record, field) == value


def test_fill_exit_calculates_net_pnl_pct_for_long_position(
    trade_record, exit_fill, matched_amounts
):
    trade_record.fill_exit(exit_fill, matched_amounts)

    expected_pnl_pct = trade_record.net_pnl / (45000.0 * 0.4995)
    assert trade_record.net_pnl_pct == pytest.approx(expected_pnl_pct, rel=1e-9)


@pytest.mark.parametrize(
    (
        "entry_price",
        "exit_price",
        "gross_quantity",
        "net_quantity",
        "entry_fee",
        "exit_fee",
        "side",
        "expected_gross",
        "expected_net",
    ),
    [
        (100.0, 110.0, 1.0, 0.99, 1.0, 1.1, "long", 10.0, 7.8),
        (100.0, 90.0, 1.0, 0.99, 1.0, 0.9, "long", -10.0, -11.8),
        (100.0, 90.0, 1.0, 0.99, 1.0, 0.9, "short", 10.0, 8.0),
        (100.0, 110.0, 1.0, 0.99, 1.0, 1.1, "short", -10.0, -12.0),
    ],
)
def test_pnl_calculations_with_various_scenarios(
    entry_price,
    exit_price,
    gross_quantity,
    net_quantity,
    entry_fee,
    exit_fee,
    side,
    expected_gross,
    expected_net,
):
    opening_fill = pd.Series(
        {
            "symbol": "BTC/USDT",
            "side": side,
            "timestamp": datetime(2024, 1, 1, 12, 0),
            "price": entry_price,
            "gross_quantity": gross_quantity,
            "net_quantity": net_quantity,
            "fee": entry_fee,
        }
    )

    closing_fill = pd.Series(
        {
            "timestamp": datetime(2024, 1, 2, 12, 0),
            "price": exit_price,
        }
    )

    matched_amounts = MatchedAmounts(
        net=net_quantity, gross=gross_quantity, entry_fee=entry_fee, exit_fee=exit_fee
    )

    trade = TradeRecord.from_entry(opening_fill)
    trade.fill_exit(closing_fill, matched_amounts)

    assert trade.gross_pnl == pytest.approx(expected_gross, rel=1e-9)
    assert trade.net_pnl == pytest.approx(expected_net, rel=1e-9)


def test_fill_exit_converts_pnl_into_account_currency(entry_fill):
    entry_fill["rate"] = 0.8
    closing_fill = pd.Series(
        {
            "timestamp": datetime(2024, 1, 2, 12, 0),
            "price": 46000.0,
            "rate": 0.5,
        }
    )
    matched_amounts = MatchedAmounts(net=0.999, gross=1.0, entry_fee=10.0, exit_fee=5.0)

    trade = TradeRecord.from_entry(entry_fill)
    trade.fill_exit(closing_fill, matched_amounts)

    assert trade.gross_pnl == pytest.approx(1000.0 * 0.5)
    assert trade.net_pnl == pytest.approx(999.0 * 0.5 - 15.0)
    assert trade.net_pnl_pct == pytest.approx(
        (999.0 * 0.5 - 15.0) / (45000.0 * 0.8 * 0.999)
    )
