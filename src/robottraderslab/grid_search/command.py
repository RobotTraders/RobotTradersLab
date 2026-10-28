import logging
import shutil
from pathlib import Path

from robottraderslab._core import setup_notebook_logging
from robottraderslab.bootstrap import BotConfig, bot_file, grid_search_file
from robottraderslab.exceptions import BotCriticalError
from robottraderslab.reporting import grid_search_folder, timestamp_now

from .grid_search_results import ERROR_COLUMN, GridSearchResults
from .optimisation_config import OptimisationConfig
from .runner import sweep

logger = logging.getLogger(__name__)

_LOG_FILENAME = "grid-search.log"
_RESULTS_FILENAME = "results.csv"


def main(folder: Path, output_csv: Path | None) -> None:
    """Score every parameter combination the bot's `grid-search.toml` declares,
    and write the run to `grid-searches/<moment>/` in the bot's folder: the
    two files it ran, its log and `results.csv`, which `output_csv` also
    receives when it is given.

    Raises:
        BotCriticalError: If the config is unusable, or if no point of
            the grid scored.
        ExchangeCriticalError: If a backtest cannot reach what it needs.
    """
    bot_path = bot_file(folder)
    search_path = grid_search_file(folder)
    search = OptimisationConfig.from_file(search_path)
    bot_config = BotConfig.from_file(bot_path)

    run_folder = grid_search_folder(folder, timestamp_now())
    run_folder.mkdir(parents=True)
    shutil.copyfile(search_path, run_folder / search_path.name)
    shutil.copyfile(bot_path, run_folder / bot_path.name)

    setup_notebook_logging(logfile=run_folder / _LOG_FILENAME)
    logger.info("Starting grid search with settings: %s", search)

    results = sweep(bot_config, search)

    written = results.to_csv(run_folder / _RESULTS_FILENAME)
    logger.info("Results exported to: %s", written)
    if output_csv is not None:
        logger.info("Results exported to: %s", results.to_csv(output_csv))

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
