import logging
from collections.abc import Iterator
from contextlib import contextmanager

from .exceptions import ExchangeRecoverableError, MissingOhlcvDataError
from .trading_system import TradingMode

logger = logging.getLogger(__name__)


@contextmanager
def profile_failure_boundary(trading_mode: TradingMode, label: str) -> Iterator[None]:
    """Contain one profile's failure so the strategy's other profiles still trade.

    Critical errors always propagate. Backtest mode surfaces bugs during
    development.

    Args:
        label: Identifies the profile in log messages.
    """
    if trading_mode is not TradingMode.LIVE:
        yield
        return
    try:
        yield
    except MissingOhlcvDataError as e:
        logger.warning("Profile %s sat out this candle: %s", label, e)
    except ExchangeRecoverableError as e:
        logger.warning("Exchange rejected profile %s this candle: %s", label, e)
    except Exception as e:
        logger.error("Internal error in profile %s this candle: %s", label, e)
