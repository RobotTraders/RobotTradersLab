import pandas as pd

from .equity_metrics_models import EquityMetricsResult


class EquityMetricsMixin:
    """The series properties hand out copies, so a reader cannot corrupt the
    measured run.
    """

    _equity_metrics: EquityMetricsResult

    @property
    def roi(self) -> float | None:
        return self._equity_metrics.roi

    @property
    def max_drawdown(self) -> float | None:
        return self._equity_metrics.max_drawdown

    @property
    def drawdown_series(self) -> pd.Series | None:
        shares = self._equity_metrics.drawdown_percentage
        return None if shares is None else shares.copy()

    @property
    def absolute_drawdown_series(self) -> pd.Series:
        return self._equity_metrics.absolute_drawdown.copy()

    @property
    def sharpe_ratio(self) -> float | None:
        return self._equity_metrics.sharpe_ratio

    @property
    def sortino_ratio(self) -> float | None:
        return self._equity_metrics.sortino_ratio

    @property
    def calmar_ratio(self) -> float | None:
        return self._equity_metrics.calmar_ratio

    @property
    def romad_ratio(self) -> float | None:
        return self._equity_metrics.return_over_max_drawdown

    @property
    def returns(self) -> pd.Series | None:
        returns = self._equity_metrics.returns
        return None if returns is None else returns.copy()

    @property
    def hodl_return(self) -> float | None:
        hodls = self._equity_metrics.hodls
        return hodls[0].hodl_return if hodls else None

    @property
    def performance_vs_hodl(self) -> float | None:
        hodls = self._equity_metrics.hodls
        return hodls[0].performance_vs_hodl if hodls else None

    @property
    def return_over_max_drawdown(self) -> float | None:
        return self._equity_metrics.return_over_max_drawdown
