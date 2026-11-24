from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from robottraderslab.reporting import grid_search_folder, timestamp_now

from .optimisation_result import SUMMARY_METRICS, OptimisationResult

ERROR_COLUMN = "error"
_NON_PARAMETER_COLUMNS = (*SUMMARY_METRICS, ERROR_COLUMN)
_SWEPT_PARAMETER_COUNT = 2
_RESULTS_FILENAME = "results.csv"


class GridSearchResults:
    """Every point of a grid search and how it scored."""

    def __init__(
        self,
        df: pd.DataFrame,
        param_columns: tuple[str, str],
        config_dir: Path | None = None,
    ) -> None:
        """Results held as a table, for a sweep already scored.

        Args:
            df: One row per grid point, holding its two parameters, the summary
                metrics and `error`.
            param_columns: The two swept parameters' columns, the first drawn
                down the side of a heatmap and the second along the bottom.
            config_dir: The bot's folder, under which `to_csv` writes when it
                is given no path.
        """
        self._df = df
        self._param_columns = param_columns
        self._config_dir = config_dir

    @classmethod
    def from_results(
        cls,
        results: list[OptimisationResult],
        config_dir: Path | None = None,
    ) -> "GridSearchResults":
        """The results of a sweep, as the grid search scored them.

        Args:
            results: One result per grid point; the first holding two
                parameters names the swept columns.
            config_dir: The bot's folder, under which `to_csv` writes when it
                is given no path.

        Raises:
            ValueError: If no result carries the two parameters a grid search
                sweeps.
        """
        df = _build_dataframe(results)
        param_columns = _extract_param_columns(results)
        return cls(df, param_columns, config_dir)

    @classmethod
    def from_csv(cls, path: str | Path) -> "GridSearchResults":
        """A file written by `to_csv` reads back into the same results.

        Args:
            path: The CSV `to_csv` wrote.

        Raises:
            FileNotFoundError: If `path` does not exist.
            ValueError: If the file lacks any column `to_csv` writes, or holds
                other than the two parameter columns a grid search sweeps.
        """
        df = pd.read_csv(path)
        return cls(df, _swept_parameter_columns(df, path))

    @property
    def dataframe(self) -> pd.DataFrame:
        """One row per grid point: the two parameter columns, the summary
        metrics and `error`, which a point that raised carries and a point
        that scored leaves empty.
        """
        p1, p2 = self._param_columns
        return self._df[[p1, p2, *SUMMARY_METRICS, ERROR_COLUMN]]

    def to_csv(self, path: str | Path | None = None) -> Path:
        """Write the scored combinations to a CSV file.

        Args:
            path: Defaults to `grid-searches/<moment>/results.csv` under
                `config_dir`, or under the working directory when the results
                carry none.

        Returns:
            The file written, which `from_csv` reads back.
        """
        if path is None:
            folder = grid_search_folder(self._config_dir or Path(), timestamp_now())
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / _RESULTS_FILENAME
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
                `risk_reward_ratio`, `sharpe_ratio`, `sortino_ratio`,
                `calmar_ratio` and `closed_trades`.
            smoothing: Sigma of the Gaussian blur applied to the grid, 0 for
                the raw values.
            title: The figure's title, the metric's display name when omitted.
            filename: Name of the saved figure, the metric's when omitted.
            save_to_path: Directory the figure is written to; omitted, the
                figure is shown.

        Raises:
            KeyError: If `metric` is not a column of the results.
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
        """Plot a grid of heatmaps, one per metric: `roi`, `win_rate`,
        `max_drawdown`, `risk_reward_ratio`, `sharpe_ratio`, `sortino_ratio`,
        `calmar_ratio` and `closed_trades`.

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
        if len(params) == _SWEPT_PARAMETER_COUNT:
            return params[0], params[1]
    raise ValueError(
        f"Grid search results must have exactly {_SWEPT_PARAMETER_COUNT} parameters"
    )


def _swept_parameter_columns(df: pd.DataFrame, path: str | Path) -> tuple[str, str]:
    missing = [c for c in _NON_PARAMETER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"`{path}` lacks the columns "
            f"{', '.join(f'`{c}`' for c in missing)} that `to_csv` writes; "
            f"run the grid search again to write it"
        )
    param_columns = [c for c in df.columns if c not in _NON_PARAMETER_COLUMNS]
    if len(param_columns) != _SWEPT_PARAMETER_COUNT:
        raise ValueError(
            f"`{path}` holds the parameter columns {param_columns}, where a "
            f"grid search sweeps {_SWEPT_PARAMETER_COUNT}"
        )
    return param_columns[0], param_columns[1]
