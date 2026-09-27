import pandas as pd

from robottraderslab._core import FRACTION_TO_PERCENT

_REASON_COLUMNS = ["Count", "% of Total", "Win Rate", "Avg PnL"]
_PAIR_KEY = "Entry Reason -> Exit Reason"
_PAIR_COLUMNS = [_PAIR_KEY, *_REASON_COLUMNS]


def calculate_reason_summary_table(
    trades_df: pd.DataFrame, reason_column: str, index_name: str = "Reason"
) -> pd.DataFrame:
    """A trade with no reason recorded is left out of the table and of its
    shares, so the shares add up to one over the trades that state a reason.

    Args:
        trades_df: Completed trades.
        reason_column: "entry_reason" or "exit_reason".
    """
    with_reason = trades_df[trades_df[reason_column].notna()]
    if with_reason.empty:
        return _empty_reason_table(index_name)
    table = _summary_by(with_reason, with_reason[reason_column])
    table.index.name = index_name
    return table


def calculate_entry_exit_pairs_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    """A trade missing either reason is left out, so every pair names both ends."""
    with_reasons = trades_df[
        trades_df["entry_reason"].notna() & trades_df["exit_reason"].notna()
    ]
    if with_reasons.empty:
        return _empty_pairs_table()
    pairs = (
        with_reasons["entry_reason"].astype(str)
        + " -> "
        + with_reasons["exit_reason"].astype(str)
    )
    return _summary_by(with_reasons, pairs).rename_axis(_PAIR_KEY).reset_index()


def _summary_by(trades: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    pnl_by_key = trades["net_pnl"].groupby(key.to_numpy())
    count = pnl_by_key.size()
    wins = trades["net_pnl"].gt(0).groupby(key.to_numpy()).sum()
    table = pd.DataFrame(
        {
            "Count": count,
            "% of Total": count / len(trades) * FRACTION_TO_PERCENT,
            "Win Rate": wins / count * FRACTION_TO_PERCENT,
            "Avg PnL": pnl_by_key.mean(),
        }
    )
    return table.sort_values("Count", ascending=False)


def _empty_reason_table(index_name: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=_REASON_COLUMNS)
    empty.index.name = index_name
    return empty


def _empty_pairs_table() -> pd.DataFrame:
    return pd.DataFrame(columns=_PAIR_COLUMNS)
