from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd

from robottraderslab._core import SymbolStr
from robottraderslab.exchanges import OrderSide, PositionSide

type FillType = Literal[
    "enter_long",
    "enter_short",
    "exit_long",
    "exit_short",
    "add_to_long",
    "reduce_long",
    "add_to_short",
    "reduce_short",
    "liquidate_long",
    "liquidate_short",
    "spot_buy",
    "spot_sell",
]


class FillRecorder(Protocol):
    def record_fill(
        self,
        timestamp: datetime,
        symbol: SymbolStr,
        side: OrderSide | PositionSide,
        gross_quantity: float,
        net_quantity: float,
        price: float,
        fee: float,
        fill_type: FillType,
        reason: str | None,
        tag: str | None = None,
        extra_fields: dict[str, Any] | None = None,
        rate: float = 1.0,
    ) -> None:
        """Records a trade fill.

        Args:
            side: Fill side (long/short)
            gross_quantity: Gross quantity in base currency before fees
            net_quantity: Net quantity after fees (actual position size)
            fee: Fee in the account currency.
            reason: Reason for this fill (e.g., "MA crossover" for entry,
                "target reached" for exit)
            tag: Label the order carried, naming the profile that placed it
            extra_fields: Custom fields to store with fill (e.g.,
                {"spread": 0.05, "rsi": 65})
            rate: Conversion rate from the quote currency into the account
                currency at fill time; `fee` is already converted.
        """


class NullFillRecorder(FillRecorder):
    def record_fill(
        self,
        timestamp: datetime,
        symbol: SymbolStr,
        side: OrderSide | PositionSide,
        gross_quantity: float,
        net_quantity: float,
        price: float,
        fee: float,
        fill_type: FillType,
        reason: str | None,
        tag: str | None = None,
        extra_fields: dict[str, Any] | None = None,
        rate: float = 1.0,
    ) -> None: ...


class FillPrinter(FillRecorder):
    def record_fill(
        self,
        timestamp: datetime,
        symbol: SymbolStr,
        side: OrderSide | PositionSide,
        gross_quantity: float,
        net_quantity: float,
        price: float,
        fee: float,
        fill_type: FillType,
        reason: str | None,
        tag: str | None = None,
        extra_fields: dict[str, Any] | None = None,
        rate: float = 1.0,
    ) -> None:
        reason_str = f"Reason: {reason}" if reason else "Reason: None"
        tag_str = f"Tag: {tag}" if tag else "Tag: None"
        extra_str = f"Extra: {extra_fields}" if extra_fields else "Extra: None"
        print(
            f"{timestamp} | "
            f"{symbol} | "
            f"{side} | "
            f"Gross: {gross_quantity} | "
            f"Net: {net_quantity} | "
            f"Price: {price} | "
            f"Fee: {fee} | "
            f"Type: {fill_type} | "
            f"{reason_str} | "
            f"{tag_str} | "
            f"{extra_str}"
        )  # pragma: no cover


class CacheFillRecorder(FillRecorder):
    """Simple fill recorder that stores fills as lists."""

    HEADER = [
        "timestamp",
        "symbol",
        "side",
        "gross_quantity",
        "net_quantity",
        "price",
        "fee",
        "fill_type",
        "reason",
        "tag",
        "extra_fields",
        "rate",
    ]

    def __init__(self) -> None:
        self._fills: list[list] = []

    @property
    def fills(self) -> list[list]:
        return deepcopy(self._fills)

    def record_fill(
        self,
        timestamp: datetime,
        symbol: SymbolStr,
        side: OrderSide | PositionSide,
        gross_quantity: float,
        net_quantity: float,
        price: float,
        fee: float,
        fill_type: FillType,
        reason: str | None,
        tag: str | None = None,
        extra_fields: dict[str, Any] | None = None,
        rate: float = 1.0,
    ) -> None:
        """Records a trade fill.

        Args:
            side: Fill side (long/short)
            gross_quantity: Gross quantity in base currency before fees
            net_quantity: Net quantity in base currency after fees
            reason: Reason for this fill (e.g., "MA crossover" for entry,
                "target reached" for exit)
            tag: Label the order carried, naming the profile that placed it
            extra_fields: Custom fields to store with fill (e.g.,
                {"spread": 0.05, "rsi": 65})
            rate: Conversion rate from the quote currency into the account
                currency at fill time; `fee` is already converted.
        """
        fill = [
            timestamp,
            symbol,
            side,
            gross_quantity,
            net_quantity,
            price,
            fee,
            fill_type,
            reason,
            tag,
            extra_fields,
            rate,
        ]
        self._fills.append(fill)

    def get_fills(self) -> pd.DataFrame:
        """Convert fills to a standardized DataFrame format for analysis.

        Extra fields are unpacked into columns with "custom_" prefix.

        Returns:
            pd.DataFrame with columns:
                - symbol: Trading pair
                - side: Fill side (long/short)
                - gross_quantity: Gross quantity in base currency before fees
                - net_quantity: Net quantity in base currency after fees
                - price: Price per unit
                - fee: Trading fee
                - fill_type: Full fill type
                - reason: Reason for fill
                - tag: Label the order carried, naming the profile that placed it
                - custom_*: Custom fields from extra_fields dict
            Index: datetime of each fill
        """
        fill_df = pd.DataFrame(self._fills, columns=self.HEADER)
        fill_df = fill_df.set_index("timestamp")

        if "extra_fields" in fill_df.columns:
            extra_df = fill_df["extra_fields"].apply(
                lambda x: pd.Series(x) if x else pd.Series(dtype=object)
            )
            extra_df = extra_df.add_prefix("custom_")

            fill_df = fill_df.drop(columns=["extra_fields"])
            fill_df = pd.concat([fill_df, extra_df], axis=1)

        return fill_df

    def to_csv(self, filename: str | Path = "fills.csv") -> None:
        """An existing file at the path is always overwritten."""
        df = self.get_fills()
        df.to_csv(str(filename))
