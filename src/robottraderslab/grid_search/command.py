import logging
from pathlib import Path

from robottraderslab._core import setup_notebook_logging
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import StrategyCriticalError

from .grid_search_results import ERROR_COLUMN, GridSearchResults
from .runner import run_grid_search

logger = logging.getLogger(__name__)


def main(config: Path, output_csv: Path | None) -> None:
    """Score every parameter combination the config declares.

    Raises:
        StrategyCriticalError: If the config is unusable, or if no point of
            the grid scored.
        ExchangeCriticalError: If a backtest cannot reach what it needs.
    """
    setup_notebook_logging(enable_file=False)
    logger.info("Starting grid search with settings: %s", config)

    results = run_grid_search(BotConfig.from_file(config))

    if output_csv:
        results.to_csv(output_csv)
        logger.info("Results exported to: %s", output_csv)

    _report_outcome(results)


def _report_outcome(results: GridSearchResults) -> None:
    """A point that fails costs its own point, so a grid keeping one scored
    point still has a surface to read.

    Raises:
        StrategyCriticalError: If no point of the grid scored.
    """
    errors = results.dataframe[ERROR_COLUMN]
    total = len(errors)
    scored = int(errors.isna().sum())

    logger.info("Grid search completed: %s/%s successful", scored, total)

    if not scored:
        raise StrategyCriticalError(
            f"No point of the {total}-point grid scored. The first failed with: "
            f"{errors.iloc[0]}"
        )
