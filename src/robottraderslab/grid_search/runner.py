import logging
from pathlib import Path

from robottraderslab._core import auto_configure_notebook_logging
from robottraderslab.bootstrap import BotConfig

from .grid_search_results import GridSearchResults
from .optimisation_config import OptimisationConfig
from .optimiser import GridSearchOptimiser

logger = logging.getLogger(__name__)


def run_grid_search(
    bot_config: BotConfig, search_file: str | Path
) -> GridSearchResults:
    """Sweep a bot over the grid a `grid-search.toml` declares and score every
    point.

    Args:
        bot_config: The bot to sweep, read from its `bot.toml`.
        search_file: The `grid-search.toml` declaring the profile to sweep,
            its parameters, their ranges and the worker count.

    Returns:
        Every point's outcome, ready to analyse and plot.

    Raises:
        BotCriticalError: If the search file is unusable, carries a section of
            the bot, picks no profile or several of `bot.toml`, or sweeps a number of
            parameters other than two.
    """
    return sweep(bot_config, OptimisationConfig.from_file(search_file))


def sweep(bot_config: BotConfig, search: OptimisationConfig) -> GridSearchResults:
    """The search as already read, for a caller that checks the file first."""
    auto_configure_notebook_logging()
    optimiser = GridSearchOptimiser(bot_config, search)
    results = optimiser.run()
    return GridSearchResults.from_results(results, config_dir=bot_config.config_dir)
