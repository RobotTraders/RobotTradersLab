from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True, slots=True)
class TradeFilter:
    """A trade is kept only when every criterion that is set matches it."""

    side: Literal["long", "short"] | None = None
    symbol: str | None = None
    profile_name_contains: str | list[str] | None = None

    def apply_filter(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df.empty:
            return trades_df.copy()

        required_columns = set()
        if self.side is not None:
            required_columns.add("side")
        if self.symbol is not None:
            required_columns.add("symbol")
        if self.profile_name_contains is not None:
            required_columns.add("profile_name")

        missing_columns = required_columns - set(trades_df.columns)
        if missing_columns:
            raise ValueError(
                f"Required columns {missing_columns} not found in trades dataframe. "
                f"Available columns: {list(trades_df.columns)}"
            )

        filtered = trades_df.copy()

        if self.side is not None:
            filtered = filtered[filtered["side"] == self.side]

        if self.symbol is not None:
            filtered = filtered[filtered["symbol"] == self.symbol]

        if self.profile_name_contains is not None:
            filtered = self._apply_profile_name_filter(filtered)

        return filtered

    def _apply_profile_name_filter(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        mask = trades_df["profile_name"].notna()
        pattern_mask = pd.Series(True, index=trades_df.index)

        for pattern in self.profile_name_patterns:
            pattern_mask &= trades_df["profile_name"].str.contains(
                pattern, case=True, na=False
            )

        return trades_df[mask & pattern_mask]

    @property
    def profile_name_patterns(self) -> list[str]:
        if self.profile_name_contains is None:
            return []
        if isinstance(self.profile_name_contains, str):
            return [self.profile_name_contains]
        return list(self.profile_name_contains)

    @property
    def is_active(self) -> bool:
        return (
            self.side is not None
            or self.symbol is not None
            or self.profile_name_contains is not None
        )

    @property
    def description(self) -> str:
        """The criteria in a short form, to label a report."""
        if not self.is_active:
            return "no_filter"

        parts = []
        if self.side is not None:
            parts.append(f"side_{self.side}")
        if self.symbol is not None:
            parts.append(f"symbol_{self.symbol.replace('/', '_').replace(':', '_')}")
        if self.profile_name_patterns:
            parts.append(f"profile_{'_and_'.join(self.profile_name_patterns)}")

        return "_".join(parts)
