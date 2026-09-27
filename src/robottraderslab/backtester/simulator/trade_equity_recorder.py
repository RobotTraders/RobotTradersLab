from .equity_snapshot import EquitySnapshot


class TradeEquityRecorder:
    """
    Records equity during backtest whenever trades occur (positions opened/closed).
    """

    def __init__(self) -> None:
        self.snapshots: list[EquitySnapshot] = []

    def record_snapshot(self, equity_snapshot: EquitySnapshot) -> None:
        """Trusts the caller to call this only when a trade just occurred."""
        self.snapshots.append(equity_snapshot.frozen_copy())
