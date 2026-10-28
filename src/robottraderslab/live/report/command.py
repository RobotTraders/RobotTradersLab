import logging
from pathlib import Path

from robottraderslab.bootstrap import BotConfig

from .runners import run_report


def main(folder: Path, days: int, save: bool) -> None:
    """Chart the account's performance over the window against its indicators.

    Raises:
        BotCriticalError: If the config is unusable.
        ExchangeCriticalError: If the account cannot be read.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_report(BotConfig.from_folder(folder), days=days, save=save)
