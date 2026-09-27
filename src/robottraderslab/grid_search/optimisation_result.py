from dataclasses import dataclass

from robottraderslab.backtester import BacktestOutputs

from .grid_point import GridPoint
from .parameter import ParameterValue

SUMMARY_METRICS = (
    "roi",
    "win_rate",
    "max_drawdown",
    "risk_reward_ratio",
    "sharpe_ratio",
    "closed_trades",
)


@dataclass(frozen=True, slots=True)
class OptimisationResult:
    grid_point: GridPoint
    error: str | None = None
    sharpe_ratio: float | None = None
    roi: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    risk_reward_ratio: float | None = None
    closed_trades: int | None = None

    @staticmethod
    def from_backtest(
        grid_point: GridPoint, backtest_outputs: BacktestOutputs
    ) -> "OptimisationResult":
        analyser = backtest_outputs.create_analyser(save=False)
        summary = analyser.get_summary_metrics()

        return OptimisationResult(
            grid_point=grid_point,
            error=None,
            sharpe_ratio=summary.sharpe_ratio,
            roi=summary.roi,
            max_drawdown=summary.max_drawdown,
            win_rate=summary.win_rate,
            risk_reward_ratio=summary.risk_reward_ratio,
            closed_trades=summary.closed_trades,
        )

    @staticmethod
    def from_error(grid_point: GridPoint, error: str) -> "OptimisationResult":
        return OptimisationResult(
            grid_point=grid_point,
            error=error,
        )

    @property
    def is_successful(self) -> bool:
        return self.error is None

    @property
    def parameters(self) -> dict[str, ParameterValue]:
        return self.grid_point.parameters
