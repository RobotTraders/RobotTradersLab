from collections.abc import Generator
from datetime import datetime
from typing import Any, ClassVar

from robottraderslab import TimeFrame
from robottraderslab._core import TimeframeSnapshot
from robottraderslab.exchanges import MarketType
from robottraderslab.strategies import BookKeeper, StrategyProtocol


class StrategyDoesntFollowProtocol: ...


class StrategyImplementsProtocol:
    """
    This class doesn't inherit from the protocol, but implements it and is
    considered a valid strategy.
    """

    market_type: ClassVar[MarketType] = "futures"

    def generate_trading_signals(self) -> None: ...

    def book_trading_actions(
        self,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: set[TimeFrame] | None = None,
    ) -> None: ...

    def iter_timeframes(
        self, start_idx: int = 0
    ) -> Generator[TimeframeSnapshot, None, None]:
        yield from ()

    async def setup(self) -> None: ...


class StrategySubclassesProtocol(StrategyProtocol):
    """
    This class inherits from the protocol and is considered a valid strategy.

    For sure this strategy won't work, but it is only for testing purposes.
    """

    market_type: ClassVar[MarketType] = "futures"


class StrategyCapturesSettings(StrategyProtocol):
    market_type: ClassVar[MarketType] = "futures"

    def __init__(self, **settings: dict[str, Any]):
        self.received_settings = settings
