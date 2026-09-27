import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any, cast

import pandas as pd

from robottraderslab._core import StepCount, to_quantity, to_step_count

from .accounting_methods import (
    AccountingMethodChoice,
    get_accounting_method,
)
from .analysis_inputs import TRADE_COLUMNS
from .trade_record import MatchedAmounts, TradeRecord

logger = logging.getLogger(__name__)

_OPENING_FILL_TYPES = ("enter", "add", "spot_buy")
_CLOSING_FILL_TYPES = ("exit", "reduce", "liquidate", "spot_sell")
_SPOT_FILL_TYPES = ("spot_buy", "spot_sell")


def create_trade_aggregation(
    fills_df: pd.DataFrame,
    accounting_method: AccountingMethodChoice = AccountingMethodChoice.FIFO,
) -> "TradeAggregator":
    aggregator = TradeAggregator(fills_df, accounting_method)
    aggregator.aggregate_trades()
    return aggregator


class TradeAggregator:
    """An exit consumes its symbol's still-open entries in the order the
    accounting method picks.
    """

    def __init__(
        self,
        fills_df: pd.DataFrame,
        accounting_method: AccountingMethodChoice = AccountingMethodChoice.FIFO,
    ) -> None:
        """
        Args:
            fills_df: One row per fill, indexed by its timestamp, with symbol,
                side, gross_quantity, net_quantity, price, fee and fill_type
                columns. Quantities are in base currency and fee in quote
                currency; net_quantity is the actual position size once fees
                have shaved gross_quantity down. fill_type names the action,
                e.g. "enter_long" or "exit_long".
        """
        _validate_fills_df(fills_df)
        self._fills = fills_df.copy()
        self._accounting_method = get_accounting_method(accounting_method)

        self._open_trades: defaultdict[tuple[str, str], list[TradeRecord]] = (
            defaultdict(list)
        )
        self._completed_trades: list[TradeRecord] = []

    @property
    def open_trades(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                asdict(trade)
                for trades in self._open_trades.values()
                for trade in trades
            ],
            columns=TRADE_COLUMNS,
        )

    @property
    def trades(self) -> pd.DataFrame:
        trades_df = pd.DataFrame(
            [asdict(trade) for trade in self._completed_trades],
            columns=TRADE_COLUMNS,
        )
        if not trades_df.empty:
            trades_df = trades_df.sort_values("entry_time", ascending=False)
        return trades_df

    def aggregate_trades(self) -> None:
        sorted_fills = self._fills.sort_index(kind="stable").reset_index(
            names="timestamp"
        )

        for fill in cast(list[dict[str, Any]], sorted_fills.to_dict("records")):
            fill_type = fill["fill_type"]
            if fill_type.startswith(_SPOT_FILL_TYPES):
                fill["side"] = "long"
            position_key = (fill["symbol"], fill["side"])

            if fill_type.startswith(_OPENING_FILL_TYPES):
                self._open_trades[position_key] = self._accounting_method.handle_entry(
                    self._open_trades[position_key], TradeRecord.from_entry(fill)
                )
            elif fill_type.startswith(_CLOSING_FILL_TYPES):
                completed, remaining = self._match_exit(
                    self._open_trades[position_key], fill
                )
                self._completed_trades.extend(completed)
                self._open_trades[position_key] = remaining

    def _match_exit(
        self, entries: list[TradeRecord], fill: Mapping[str, Any]
    ) -> tuple[list[TradeRecord], list[TradeRecord]]:
        """Every quantity that entered leaves in exactly one trade, and the entries
        still open keep their fill order so an accounting method reads the same
        history at every exit.

        No fill is finer than the quantity step an exit counts its entries down
        in, so an exit its open entries cover leaves nothing behind.

        Returns:
            The trades the exit closed, and the entries still open after it.
        """
        exit_steps = to_step_count(fill["net_quantity"])
        remaining_steps = exit_steps
        completed: list[TradeRecord] = []
        consumed: set[int] = set()

        for entry in self._accounting_method.order_trades_for_exit(list(entries)):
            if remaining_steps == 0:
                break

            entry_steps = to_step_count(entry.net_quantity)
            matched_steps = min(entry_steps, remaining_steps)
            remaining_steps -= matched_steps
            matched = _matched_amounts(
                entry, entry_steps, fill, exit_steps, matched_steps
            )

            if matched_steps < entry_steps:
                closed = replace(entry)
                _reduce_entry(entry, matched)
            else:
                closed = entry
                consumed.add(id(entry))
            closed.fill_exit(fill, matched)
            completed.append(closed)

        if remaining_steps > 0:
            logger.error(
                "%s %s: an exit of %s at %s found no open entry to close; "
                "the trade table leaves it out",
                fill["symbol"],
                fill["side"],
                to_quantity(remaining_steps),
                fill["timestamp"],
            )
        return completed, [entry for entry in entries if id(entry) not in consumed]


def _matched_amounts(
    entry: TradeRecord,
    entry_steps: StepCount,
    fill: Mapping[str, Any],
    exit_steps: StepCount,
    matched_steps: StepCount,
) -> MatchedAmounts:
    entry_share = matched_steps / entry_steps
    exit_share = matched_steps / exit_steps
    return MatchedAmounts(
        net=entry.net_quantity * entry_share,
        gross=entry.gross_quantity * entry_share,
        entry_fee=entry.entry_fee * entry_share,
        exit_fee=fill["fee"] * exit_share,
    )


def _reduce_entry(entry: TradeRecord, matched: MatchedAmounts) -> None:
    entry.net_quantity -= matched.net
    entry.gross_quantity -= matched.gross
    entry.entry_fee -= matched.entry_fee


def _validate_fills_df(fills_df: pd.DataFrame) -> None:
    required_columns = {
        "symbol",
        "side",
        "gross_quantity",
        "net_quantity",
        "price",
        "fee",
        "fill_type",
    }
    missing_columns = required_columns - set(fills_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in fills dataframe: {missing_columns}"
        )
