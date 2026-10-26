from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SummaryMetrics:
    sharpe_ratio: float | None
    sortino_ratio: float | None
    calmar_ratio: float | None
    roi: float | None
    max_drawdown: float | None
    win_rate: float
    risk_reward_ratio: float
    closed_trades: int
