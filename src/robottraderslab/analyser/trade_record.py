from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import pandas as pd

from robottraderslab._core import (
    calculate_pnl_long,
    calculate_pnl_short,
    calculate_trade_pnl_pct,
)


class MatchedAmounts(NamedTuple):
    """The share of an entry one exit fill closes, and the fees that share carries."""

    net: float
    gross: float
    entry_fee: float
    exit_fee: float


@dataclass(slots=True, eq=False)
class TradeRecord:
    """One trade, or the still-open part of one.

    Two records are two trades however alike their fields, so a record is
    equal to itself alone.
    """

    symbol: str
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    gross_quantity: float
    net_quantity: float
    entry_fee: float
    entry_rate: float = 1.0
    exit_time: pd.Timestamp | None = None
    exit_price: float = 0.0
    exit_fee: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    net_pnl_pct: float = 0.0
    entry_reason: str | None = None
    exit_reason: str | None = None
    tag: str | None = None

    @classmethod
    def from_entry(cls, fill: Mapping[str, Any]) -> "TradeRecord":
        return cls(
            symbol=fill["symbol"],
            side=fill["side"],
            entry_time=fill["timestamp"],
            entry_price=fill["price"],
            gross_quantity=fill["gross_quantity"],
            net_quantity=fill["net_quantity"],
            entry_fee=fill["fee"],
            entry_rate=fill.get("rate", 1.0),
            entry_reason=fill.get("reason"),
            tag=fill.get("tag"),
        )

    def fill_exit(self, fill: Mapping[str, Any], matched: MatchedAmounts) -> None:
        """Price PnL is converted into the account currency at the exit fill's
        rate, matching how the exchange books it.

        Fees are already converted.
        """
        self.exit_time = fill["timestamp"]
        self.exit_price = fill["price"]
        self.gross_quantity = matched.gross
        self.net_quantity = matched.net
        self.entry_fee = matched.entry_fee
        self.exit_fee = matched.exit_fee
        self.exit_reason = fill.get("reason")
        exit_rate = fill.get("rate", 1.0)

        self.gross_pnl = (
            calculate_pnl_long(self.entry_price, self.exit_price, self.gross_quantity)
            if self.side == "long"
            else calculate_pnl_short(
                self.entry_price, self.exit_price, self.gross_quantity
            )
        ) * exit_rate

        self.net_pnl = (
            (
                calculate_pnl_long(self.entry_price, self.exit_price, self.net_quantity)
                if self.side == "long"
                else calculate_pnl_short(
                    self.entry_price, self.exit_price, self.net_quantity
                )
            )
            * exit_rate
            - self.entry_fee
            - self.exit_fee
        )

        self.net_pnl_pct = calculate_trade_pnl_pct(
            self.net_pnl, self.entry_price * self.entry_rate, self.net_quantity
        )
