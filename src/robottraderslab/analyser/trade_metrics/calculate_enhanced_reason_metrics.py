from dataclasses import dataclass

import pandas as pd

from .reason_analysis import (
    calculate_entry_exit_pairs_table,
    calculate_reason_summary_table,
)


@dataclass
class ReasonBreakdown:
    """Every table counts only the trades that named the reasons it groups by."""

    entry_reasons: pd.Series
    exit_reasons: pd.Series
    entry_reasons_table: pd.DataFrame
    exit_reasons_table: pd.DataFrame
    entry_exit_pairs_table: pd.DataFrame


def calculate_enhanced_reason_metrics(
    trades_df: pd.DataFrame,
) -> ReasonBreakdown:
    entry_reasons = _reason_counts(trades_df, "entry_reason")
    exit_reasons = _reason_counts(trades_df, "exit_reason")

    entry_reasons_table = calculate_reason_summary_table(
        trades_df, "entry_reason", "Entry Reason"
    )
    exit_reasons_table = calculate_reason_summary_table(
        trades_df, "exit_reason", "Exit Reason"
    )
    entry_exit_pairs_table = calculate_entry_exit_pairs_table(trades_df)

    return ReasonBreakdown(
        entry_reasons=entry_reasons,
        exit_reasons=exit_reasons,
        entry_reasons_table=entry_reasons_table,
        exit_reasons_table=exit_reasons_table,
        entry_exit_pairs_table=entry_exit_pairs_table,
    )


def _reason_counts(trades_df: pd.DataFrame, reason_column: str) -> pd.Series:
    if reason_column in trades_df.columns:
        counts = trades_df[reason_column].value_counts()
    else:
        counts = pd.Series(dtype=int)

    counts.name = reason_column
    counts.index.name = None
    return counts
