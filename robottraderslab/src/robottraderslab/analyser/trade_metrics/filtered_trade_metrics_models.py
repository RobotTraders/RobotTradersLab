from dataclasses import dataclass

import pandas as pd

from ..trade_filter import TradeFilter


@dataclass(frozen=True, slots=True)
class FilteredTradeMetricsResult:
    """Trade-level metrics computed on trades filtered by side or symbol.

    No field reads the equity curve, which a trade filter does not narrow
    to match.
    """

    total_trades: int
    winning_trades: int
    losing_trades: int

    win_rate: float
    profit_factor: float
    risk_reward_ratio: float

    max_win_streak: int
    max_lose_streak: int

    total_fee: float
    biggest_fee: float
    avg_fee: float

    avg_trade_pnl: float
    avg_trade_return: float
    avg_winning_trade_pnl: float
    avg_losing_trade_pnl: float
    avg_winning_trade_return: float
    avg_losing_trade_return: float

    largest_winning_trade_pnl: float
    largest_losing_trade_pnl: float

    total_pnl: float

    avg_trade_duration_days: float
    avg_winning_trade_duration_days: float
    avg_losing_trade_duration_days: float

    time_in_position_ratio: float
    avg_trades_per_day: float

    open_reasons: pd.Series
    close_reasons: pd.Series

    filter_applied: TradeFilter

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        lines = [
            f"Filter: {self.filter_applied.description}",
            f"Total Trades: {self.total_trades}",
            f"Win Rate: {self.win_rate:.2%}",
            f"Profit Factor: {self.profit_factor:.2f}",
            f"Risk-Reward Ratio: {self.risk_reward_ratio:.2f}",
            f"Total PnL: ${self.total_pnl:.2f}",
            f"Avg Trade PnL: {self.avg_trade_pnl:.4f}",
            f"Avg Win PnL: {self.avg_winning_trade_pnl:.4f}",
            f"Avg Loss PnL: {self.avg_losing_trade_pnl:.4f}",
            f"Max Win Streak: {self.max_win_streak}",
            f"Max Lose Streak: {self.max_lose_streak}",
            f"Avg Trade Duration: {self.avg_trade_duration_days:.2f} days",
            f"Time in Position: {self.time_in_position_ratio:.2%}",
            f"Total Fees: ${self.total_fee:.2f}",
        ]
        return "\n".join(lines)

    def _repr_text_(self) -> str:
        return str(self)
