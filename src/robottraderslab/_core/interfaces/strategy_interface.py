from datetime import datetime
from typing import ClassVar, Protocol, runtime_checkable

from ..account_snapshot import AccountSnapshots
from ..actions.bookkeeper import BookKeeper
from ..market import MarketType
from ..ohlcv import OHLCVs
from ..strategy_requirements import StrategyRequirements
from ..timeframes import TimeFrame


@runtime_checkable
class StrategyProtocol(Protocol):
    """The three phases the engine runs a strategy through, in order.

    `setup` declares what the strategy needs, `generate_trading_signals`
    turns the candles into signals, and `book_trading_actions` books what
    should happen on the exchange. The engine constructs the strategy as
    `cls(account=..., trading_system=..., config_dir=..., **settings)`, the
    settings being the `[strategy]` keys of the configuration file;
    `ProfileStrategy` takes them for a strategy written in profiles.

    Attributes:
        market_type: The market the strategy trades, which the engine
            builds the strategy's account for.
    """

    market_type: ClassVar[MarketType]

    async def setup(self, requirements: StrategyRequirements) -> None:
        """Declare what the strategy needs, before any data is fetched.

        Runs before the first candle in a backtest and at the start of every
        live cycle, retried on a transient error. The strategy is constructed
        before this runs, outside any retry, and a failing `__init__` stops
        the run at load, so a check that can fail belongs here.

        Args:
            requirements: Where the candles, the account state and the
                trackers are declared.
        """

    def generate_trading_signals(self, ohlcvs: OHLCVs) -> None:
        """Turn the declared candles into the strategy's signals.

        Runs once before the candles are walked: over the whole window in a
        backtest, over the window fetched for the cycle in live trading. A
        signal computed here over a whole series is stored as a column with
        `add_column`, where the per-candle phase reads it with `current` and
        `signal`.

        Args:
            ohlcvs: The declared candles, the lookback ahead of the window.
        """

    def book_trading_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: list[TimeFrame],
    ) -> None:
        """Book what should happen on the venue at the moment candles closed.

        Runs once per moment a declared timeframe closes a candle, in a
        backtest and in live trading, after the account snapshots of that
        moment were taken. The actions booked run once the method returns;
        the method itself sends nothing, and a snapshot serves the declared
        state from memory.

        Args:
            ohlcvs: The declared candles and the strategy's signal columns.
            account_snapshots: The declared state of every declared account
                at this moment, read with `of(account)`.
            timestamp: The moment the candles closed.
            bookkeeper: Where the actions are booked.
            triggered_timeframes: The timeframes whose candle closed at this
                moment.
        """
