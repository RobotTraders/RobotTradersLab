from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .optimisation_result import SUMMARY_METRICS, OptimisationResult

ERROR_COLUMN = "error"
_NON_PARAMETER_COLUMNS = set(SUMMARY_METRICS) | {ERROR_COLUMN}


class GridSearchResults:
    """Every point of a grid search and how it scored, as a table, a CSV and
    heatmaps over the two parameters swept.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        param_columns: tuple[str, str],
        config_dir: Path | None = None,
    ) -> None:
        self._df = df
        self._param_columns = param_columns
        self._config_dir = config_dir

    @classmethod
    def from_results(
        cls,
        results: list[OptimisationResult],
        config_dir: Path | None = None,
    ) -> "GridSearchResults":
        df = _build_dataframe(results)
        param_columns = _extract_param_columns(results)
        return cls(df, param_columns, config_dir)

    @classmethod
    def from_csv(cls, path: str | Path) -> "GridSearchResults":
        """A file written by `to_csv` reads back into the same results.

        Raises:
            ValueError: If the file is not one `to_csv` wrote.
        """
        df = pd.read_csv(path)
        if ERROR_COLUMN not in df.columns:
            raise ValueError(
                f"`{path}` holds no `{ERROR_COLUMN}` column, so it is not a "
                f"results file `to_csv` wrote"
            )
        param_columns = [c for c in df.columns if c not in _NON_PARAMETER_COLUMNS]
        return cls(df, (param_columns[0], param_columns[1]))

    @property
    def dataframe(self) -> pd.DataFrame:
        """One row per grid point: the two parameter columns, the summary
        metrics and `error`, which a point that raised carries and a point
        that scored leaves empty.
        """
        p1, p2 = self._param_columns
        return self._df[[p1, p2, *SUMMARY_METRICS, ERROR_COLUMN]]

    def to_csv(self, path: str | Path | None = None) -> Path:
        """Export results to CSV.

        Args:
            path: Defaults to `{config_dir}/grid_search/results.csv` when omitted.

        Returns:
            Path to the created file.
        """
        if path is None:
            grid_dir = (self._config_dir or Path()) / "grid_search"
            grid_dir.mkdir(parents=True, exist_ok=True)
            path = grid_dir / "results.csv"
        else:
            path = Path(path)
        self.dataframe.to_csv(path, index=False)
        return path

    def plot_heatmap(
        self,
        metric: str,
        *,
        smoothing: float = 0,
        title: str | None = None,
        filename: str | None = None,
        save_to_path: Path | None = None,
    ) -> None:
        """Plot one metric as a heatmap across the parameter grid.

        Args:
            metric: One of `roi`, `win_rate`, `max_drawdown`,
                `risk_reward_ratio`, `sharpe_ratio` and `closed_trades`.
            smoothing: Sigma of the Gaussian blur applied to the grid, 0 for
                the raw values.
            title: The figure's title, the metric's name when omitted.
            filename: Name of the saved figure, the metric's when omitted.
            save_to_path: Directory the figure is written to; omitted, the
                figure is shown.
        """
        from .plots import plot_heatmap

        p1, p2 = self._param_columns
        plot_heatmap(
            self._df,
            p1,
            p2,
            metric,
            smoothing=smoothing,
            title=title,
            filename=filename,
            save_to_path=save_to_path,
        )

    def plot_heatmap_summary(
        self,
        *,
        smoothing: float = 0,
        title: str | None = None,
        filename: str | None = None,
        save_to_path: Path | None = None,
    ) -> None:
        """Plot a grid of heatmaps, one per summary metric.

        Args:
            smoothing: Sigma of the Gaussian blur applied to each grid, 0 for
                the raw values.
            title: The figure's title, none when omitted.
            filename: Name of the saved figure, `heatmap_summary` when
                omitted.
            save_to_path: Directory the figure is written to; omitted, the
                figure is shown.
        """
        from .plots import plot_heatmap_summary

        p1, p2 = self._param_columns
        plot_heatmap_summary(
            self._df,
            p1,
            p2,
            smoothing=smoothing,
            title=title,
            filename=filename,
            save_to_path=save_to_path,
        )


def _build_dataframe(results: list[OptimisationResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        row = asdict(result)
        row.pop("grid_point")
        row.update(result.parameters)
        rows.append(row)
    return pd.DataFrame(rows)


def _extract_param_columns(results: list[OptimisationResult]) -> tuple[str, str]:
    for result in results:
        params = list(result.parameters.keys())
        if len(params) == 2:
            return params[0], params[1]
    raise ValueError("Grid search results must have exactly 2 parameters")
