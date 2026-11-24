from datetime import datetime

from .equity_snapshot import EquitySnapshot


class DailyEquityRecorder:
    """Records equity during a backtest at every midnight and at the run's
    two bounds.
    """

    def __init__(self) -> None:
        self._snapshots: list[EquitySnapshot] = []

    @property
    def snapshots(self) -> list[EquitySnapshot]:
        return self._snapshots

    def record_bound(self, equity_snapshot: EquitySnapshot) -> None:
        """Keeps the snapshot a run opens or closes on whatever its time, so
        the curve spans the whole run and ends on the run's final state. The
        bound takes the place of a midnight point on the same moment, since
        it also counts the orders placed at that close.
        """
        if (
            self._snapshots
            and self._snapshots[-1].timestamp == equity_snapshot.timestamp
        ):
            self._snapshots[-1] = equity_snapshot.frozen_copy()
            return
        self._snapshots.append(equity_snapshot.frozen_copy())

    def record_snapshot(self, equity_snapshot: EquitySnapshot) -> None:
        """Keeps the snapshot only when its timestamp lands on midnight."""
        if _is_midnight(equity_snapshot.timestamp):
            self._snapshots.append(equity_snapshot.frozen_copy())


def _is_midnight(timestamp: datetime) -> bool:
    return timestamp.hour == 0 and timestamp.minute == 0 and timestamp.second == 0
