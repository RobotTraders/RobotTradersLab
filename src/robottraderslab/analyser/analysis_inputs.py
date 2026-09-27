from dataclasses import dataclass

import pandas as pd

TRADE_COLUMNS = (
    "entry_time",
    "exit_time",
    "symbol",
    "side",
    "tag",
    "entry_price",
    "exit_price",
    "gross_quantity",
    "net_quantity",
    "entry_fee",
    "exit_fee",
    "gross_pnl",
    "net_pnl",
    "net_pnl_pct",
    "entry_reason",
    "exit_reason",
)


@dataclass
class AnalysisInputs:
    """`trades` and `open_positions` both carry exactly `TRADE_COLUMNS`,
    whichever source built them and however many rows they hold, so a
    consumer reads one shape.

    `initial_balance` is what the equity curve is measured from. A backtest
    knows it: its curve opens on the balance the account was configured with.
    A live report recovers it from what the venue states the account holds
    now, less the profit realised across the window, and is left without one
    where the venue states no balance to take that profit off. Every ratio
    that divides by that balance is stated only when it is given.
    """

    trades: pd.DataFrame
    open_positions: pd.DataFrame
    equity_curve: pd.Series
    initial_balance: float | None = None
    reference_price: pd.Series | None = None
    reference_symbol_str: str | None = None
