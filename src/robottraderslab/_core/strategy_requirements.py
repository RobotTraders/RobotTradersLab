from .account_requirements import AccountRequirements
from .ohlcv.ohlcv_requirements import OHLCVRequirements
from .tracker_requirements import TrackerRequirements


class StrategyRequirements:
    """What a strategy declares in `setup`.

    Attributes:
        ohlcv: The candles the strategy reads, per symbol and timeframe.
        account: The account state the strategy reads on every candle.
        tracker: The positions the strategy keeps per tracking id.
    """

    def __init__(self) -> None:
        self.ohlcv = OHLCVRequirements()
        self.account = AccountRequirements()
        self.tracker = TrackerRequirements(
            self.account._require_positions,
            self.account._require_executions,
        )
