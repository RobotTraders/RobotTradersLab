import logging

from robottraderslab._core import auto_configure_notebook_logging
from robottraderslab.bootstrap import BotConfig

from .grid_search_results import GridSearchResults
from .optimiser import GridSearchOptimiser

logger = logging.getLogger(__name__)


def run_grid_search(bot_config: BotConfig) -> GridSearchResults:
    """Sweep a bot over its grid and score every point.

    Returns:
        Every point's outcome, ready to analyse and plot.

    Raises:
        StrategyCriticalError: If the grid sweeps a number of parameters other
            than two.
    """
    auto_configure_notebook_logging()
    optimiser = GridSearchOptimiser(bot_config)
    results = optimiser.run()
    return GridSearchResults.from_results(results, config_dir=bot_config.config_dir)
