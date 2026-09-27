from collections.abc import Mapping

import pandas as pd

from robottraderslab._core import PositionSnapshot, Symbol

from .analysis_inputs import TRADE_COLUMNS


def build_open_positions(positions: Mapping[Symbol, PositionSnapshot]) -> pd.DataFrame:
    """Carry the venue's open positions in the shape a closed trade would take.

    A position still open has no exit yet, so every field an exit would fill
    in carries the same default a fresh entry does: zero for amounts, absent
    for the reason. The fee paid opening it is not part of what a venue
    states about an open position, so it is stated as zero.
    """
    rows = [_row(position) for position in positions.values()]
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def _row(position: PositionSnapshot) -> dict[str, object]:
    return {
        "entry_time": pd.Timestamp(position.entry_time),
        "exit_time": None,
        "symbol": str(position.symbol),
        "side": str(position.side),
        "tag": None,
        "entry_price": position.average_entry_price,
        "exit_price": 0.0,
        "gross_quantity": position.quantity,
        "net_quantity": position.quantity,
        "entry_fee": 0.0,
        "exit_fee": 0.0,
        "gross_pnl": 0.0,
        "net_pnl": 0.0,
        "net_pnl_pct": 0.0,
        "entry_reason": None,
        "exit_reason": None,
    }
