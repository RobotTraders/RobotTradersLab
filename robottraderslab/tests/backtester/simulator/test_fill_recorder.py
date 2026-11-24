from datetime import datetime, timedelta

import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import CacheFillRecorder
from robottraderslab.exchanges import PositionSide

NOW = datetime.fromisoformat("2023-01-01")


@pytest.fixture
def fill_cache_recorder() -> CacheFillRecorder:
    return CacheFillRecorder()


@pytest.fixture
def sample_transaction() -> dict[str, object]:
    return {
        "timestamp": NOW,
        "symbol": str(Symbol.create("BTC/USDT:USDT")),
        "side": PositionSide.LONG,
        "gross_quantity": 10.0,
        "net_quantity": 9.8,
        "price": 50000.0,
        "fee": 10.0,
        "fill_type": "enter_long",
    }


def test_fill_cache_recorder_starts_empty(fill_cache_recorder):
    assert len(fill_cache_recorder.fills) == 0


def test_fill_cache_recorder_stores_transaction_in_correct_order(
    fill_cache_recorder, sample_transaction
):
    fill_cache_recorder.record_fill(
        sample_transaction["timestamp"],
        sample_transaction["symbol"],
        PositionSide.LONG,
        sample_transaction["gross_quantity"],
        sample_transaction["net_quantity"],
        sample_transaction["price"],
        sample_transaction["fee"],
        sample_transaction["fill_type"],
        reason=None,
    )

    transaction = fill_cache_recorder.fills[0]
    expected_order = [
        sample_transaction["timestamp"],
        sample_transaction["symbol"],
        sample_transaction["side"],
        sample_transaction["gross_quantity"],
        sample_transaction["net_quantity"],
        sample_transaction["price"],
        sample_transaction["fee"],
        sample_transaction["fill_type"],
        None,  # reason
        None,  # tag
        None,  # extra_fields
        1.0,  # rate
    ]
    assert transaction == expected_order


def test_fill_cache_recorder_transactions_property_returns_deep_copy(
    fill_cache_recorder, sample_transaction
):
    fill_cache_recorder.record_fill(
        sample_transaction["timestamp"],
        sample_transaction["symbol"],
        PositionSide.LONG,
        sample_transaction["gross_quantity"],
        sample_transaction["net_quantity"],
        sample_transaction["price"],
        sample_transaction["fee"],
        sample_transaction["fill_type"],
        reason=None,
    )

    transactions = fill_cache_recorder.fills
    transactions[0][0] = datetime.fromisoformat("2025-05-05 01:02:04")
    assert fill_cache_recorder.fills[0][0] == sample_transaction["timestamp"]


def test_fill_cache_recorder_maintains_transaction_chronological_order(
    fill_cache_recorder, sample_transaction
):
    timestamps = [NOW + timedelta(seconds=i) for i in range(3)]

    for timestamp in timestamps:
        tx = sample_transaction.copy()
        tx["timestamp"] = timestamp
        fill_cache_recorder.record_fill(
            tx["timestamp"],
            tx["symbol"],
            PositionSide.LONG,
            tx["gross_quantity"],
            tx["net_quantity"],
            tx["price"],
            tx["fee"],
            tx["fill_type"],
            reason=None,
        )

    recorded_timestamps = [tx[0] for tx in fill_cache_recorder.fills]
    assert recorded_timestamps == timestamps


def test_fill_cache_recorder_dataframe_contains_all_transaction_fields(
    fill_cache_recorder, sample_transaction
):
    fill_cache_recorder.record_fill(
        sample_transaction["timestamp"],
        sample_transaction["symbol"],
        PositionSide.LONG,
        sample_transaction["gross_quantity"],
        sample_transaction["net_quantity"],
        sample_transaction["price"],
        sample_transaction["fee"],
        sample_transaction["fill_type"],
        reason=None,
    )

    df = fill_cache_recorder.get_fills()
    row = df.iloc[0]

    assert df.index[0] == sample_transaction["timestamp"]
    assert row["symbol"] == sample_transaction["symbol"]
    assert row["side"] == sample_transaction["side"]
    assert row["gross_quantity"] == sample_transaction["gross_quantity"]
    assert row["net_quantity"] == sample_transaction["net_quantity"]
    assert row["price"] == sample_transaction["price"]
    assert row["fee"] == sample_transaction["fee"]
    assert row["fill_type"] == sample_transaction["fill_type"]


@pytest.mark.parametrize(
    "reason",
    [
        "MA crossover",
        None,
    ],
)
def test_fill_cache_recorder_stores_reason(
    fill_cache_recorder, sample_transaction, reason
):
    fill_cache_recorder.record_fill(
        sample_transaction["timestamp"],
        sample_transaction["symbol"],
        PositionSide.LONG,
        sample_transaction["gross_quantity"],
        sample_transaction["net_quantity"],
        sample_transaction["price"],
        sample_transaction["fee"],
        sample_transaction["fill_type"],
        reason=reason,
    )

    transaction = fill_cache_recorder.fills[0]
    assert transaction[8] == reason

    df = fill_cache_recorder.get_fills()
    assert df.iloc[0]["reason"] == reason


def test_records_transaction_with_extra_fields(fill_cache_recorder, sample_transaction):
    extra_fields = {"spread": 0.05, "rsi": 65.2, "volume_ratio": 1.8}

    fill_cache_recorder.record_fill(
        sample_transaction["timestamp"],
        sample_transaction["symbol"],
        PositionSide.LONG,
        sample_transaction["gross_quantity"],
        sample_transaction["net_quantity"],
        sample_transaction["price"],
        sample_transaction["fee"],
        sample_transaction["fill_type"],
        reason="MA crossover",
        extra_fields=extra_fields,
    )

    df = fill_cache_recorder.get_fills()

    assert len(df) == 1
    assert "custom_spread" in df.columns
    assert "custom_rsi" in df.columns
    assert "custom_volume_ratio" in df.columns
    assert df.iloc[0]["custom_spread"] == 0.05
    assert df.iloc[0]["custom_rsi"] == 65.2
    assert df.iloc[0]["custom_volume_ratio"] == 1.8


def test_records_transaction_without_extra_fields(
    fill_cache_recorder, sample_transaction
):
    fill_cache_recorder.record_fill(
        sample_transaction["timestamp"],
        sample_transaction["symbol"],
        PositionSide.LONG,
        sample_transaction["gross_quantity"],
        sample_transaction["net_quantity"],
        sample_transaction["price"],
        sample_transaction["fee"],
        sample_transaction["fill_type"],
        reason="MA crossover",
        extra_fields=None,
    )

    df = fill_cache_recorder.get_fills()

    assert len(df) == 1
    custom_cols = [col for col in df.columns if col.startswith("custom_")]
    assert len(custom_cols) == 0


def test_flattens_different_extra_fields_across_transactions(
    fill_cache_recorder, sample_transaction
):
    fill_cache_recorder.record_fill(
        sample_transaction["timestamp"],
        sample_transaction["symbol"],
        PositionSide.LONG,
        sample_transaction["gross_quantity"],
        sample_transaction["net_quantity"],
        sample_transaction["price"],
        sample_transaction["fee"],
        sample_transaction["fill_type"],
        reason="Entry",
        extra_fields={"spread": 0.05, "rsi": 65.2},
    )

    fill_cache_recorder.record_fill(
        sample_transaction["timestamp"] + timedelta(hours=1),
        sample_transaction["symbol"],
        PositionSide.LONG,
        10.0,
        9.99,
        3000.0,
        30.0,
        "enter_long",
        reason="Entry",
        extra_fields={"volatility": 0.23},
    )

    fill_cache_recorder.record_fill(
        sample_transaction["timestamp"] + timedelta(hours=2),
        sample_transaction["symbol"],
        PositionSide.LONG,
        0.5,
        0.4995,
        51000.0,
        25.5,
        "exit_long",
        reason="Exit",
        extra_fields=None,
    )

    df = fill_cache_recorder.get_fills()

    assert len(df) == 3
    assert "custom_spread" in df.columns
    assert "custom_rsi" in df.columns
    assert "custom_volatility" in df.columns
    assert df.iloc[0]["custom_spread"] == 0.05
    assert df.iloc[0]["custom_rsi"] == 65.2
    assert pd.isna(df.iloc[0]["custom_volatility"])
    assert pd.isna(df.iloc[1]["custom_spread"])
    assert pd.isna(df.iloc[1]["custom_rsi"])
    assert df.iloc[1]["custom_volatility"] == 0.23
    assert pd.isna(df.iloc[2]["custom_spread"])
    assert pd.isna(df.iloc[2]["custom_rsi"])
    assert pd.isna(df.iloc[2]["custom_volatility"])


def test_handles_empty_extra_fields_dict(fill_cache_recorder, sample_transaction):
    fill_cache_recorder.record_fill(
        sample_transaction["timestamp"],
        sample_transaction["symbol"],
        PositionSide.LONG,
        sample_transaction["gross_quantity"],
        sample_transaction["net_quantity"],
        sample_transaction["price"],
        sample_transaction["fee"],
        sample_transaction["fill_type"],
        reason="Entry",
        extra_fields={},
    )

    df = fill_cache_recorder.get_fills()

    assert len(df) == 1
    custom_cols = [col for col in df.columns if col.startswith("custom_")]
    assert len(custom_cols) == 0


def test_prefixes_extra_fields_to_avoid_column_conflicts(
    fill_cache_recorder, sample_transaction
):
    fill_cache_recorder.record_fill(
        sample_transaction["timestamp"],
        sample_transaction["symbol"],
        PositionSide.LONG,
        sample_transaction["gross_quantity"],
        sample_transaction["net_quantity"],
        sample_transaction["price"],
        sample_transaction["fee"],
        sample_transaction["fill_type"],
        reason="Entry",
        extra_fields={"symbol": "my_custom_symbol", "price": 123.45},
    )

    df = fill_cache_recorder.get_fills()

    assert len(df) == 1
    assert df.iloc[0]["symbol"] == sample_transaction["symbol"]
    assert df.iloc[0]["price"] == sample_transaction["price"]
    assert df.iloc[0]["custom_symbol"] == "my_custom_symbol"
    assert df.iloc[0]["custom_price"] == 123.45
