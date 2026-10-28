import logging
from pathlib import Path

from robottraderslab._core import setup_notebook_logging
from robottraderslab.bootstrap import BotConfig, optimisation_file
from robottraderslab.exceptions import BotCriticalError

from .grid_search_results import ERROR_COLUMN, GridSearchResults
from .runner import run_grid_search

logger = logging.getLogger(__name__)


def main(folder: Path, output_csv: Path | None) -> None:
    """Score every parameter combination the bot's `optimisation.toml` declares,
    and write the scores to `output_csv`, or to `optimisations/<moment>/` in the
    bot's folder when it is None.

    Raises:
        BotCriticalError: If the config is unusable, or if no point of
            the grid scored.
        ExchangeCriticalError: If a backtest cannot reach what it needs.
    """
    config = optimisation_file(folder)
    setup_notebook_logging(enable_file=False)
    logger.info("Starting grid search with settings: %s", config)

    results = run_grid_search(BotConfig.from_file(config))

    written = results.to_csv(output_csv)
    logger.info("Results exported to: %s", written)

    _report_outcome(results)


def _report_outcome(results: GridSearchResults) -> None:
    """A point that fails costs its own point, so a grid keeping one scored
    point still has a surface to read.

    Raises:
        BotCriticalError: If no point of the grid scored.
    """
    errors = results.dataframe[ERROR_COLUMN]
    total = len(errors)
    scored = int(errors.isna().sum())

    logger.info("Grid search completed: %s/%s successful", scored, total)

    if not scored:
        raise BotCriticalError(
            f"No point of the {total}-point grid scored. The first failed with: "
            f"{errors.iloc[0]}"
        )
