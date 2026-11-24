from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar

from robottraderslab import Symbol, TimeFrame
from robottraderslab.exceptions import BotCriticalError
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    MarketType,
    OHLCVs,
    PositionTracker,
    ProfileStrategy,
    StrategyRequirements,
    SymbolTimeframe,
    TrackingId,
)
from robottraderslab.strategies.futures import FuturesAccount, MarginSettings

from .impulse_indicator import compute_signals
from .position_orders import book_entry, book_exit, wanted_entry_side, wants_exit
from .position_protection import book_position_protections
from .position_tracking import tags_by_profile_id, tracked_symbols
from .profile_config import ProfileConfig
from .profile_snapshot import ProfileSnapshot


class ImpulseStrategy(ProfileStrategy[ProfileConfig]):
    """Momentum strategy: enters on a TRIX crossover that agrees with the trend.

    Each profile covers one symbol on one timeframe. Profiles may stack on the
    same symbol, each opening and closing its own position independently.
    """

    market_type: ClassVar[MarketType] = "futures"

    account: FuturesAccount
    _position_tracker: PositionTracker
    _closing_profile_ids: set[TrackingId]

    async def setup(self, requirements: StrategyRequirements) -> None:
        """
        Raises:
            BotCriticalError: If a profile's tag is too long, or two
                profiles on the same symbol resolve to the same tag or disagree
                on stop_loss_pct, leverage or margin_mode.
        """
        _validate_stop_loss_agreement(self.profiles)
        margin_targets = _margin_targets(self.profiles)

        lookbacks = _lookbacks_by_timeframe(self.profiles)
        for symbol_timeframe in _symbol_timeframes(self.profiles):
            requirements.ohlcv.add(
                symbol_timeframe.symbol,
                symbol_timeframe.timeframe,
                lookbacks[symbol_timeframe.timeframe],
            )

        requirements.account.add(
            self.account,
            symbols=[profile.symbol for profile in self.profiles],
            positions=True,
            open_orders=True,
            balances=True,
            sizing=[profile.sizing for profile in self.profiles],
            margin_targets=margin_targets,
        )
        self._position_tracker = requirements.tracker.add(
            account=self.account,
            symbols=tracked_symbols(self.profiles),
            tags=tags_by_profile_id(self.profiles),
        )

    def generate_profile_signals(self, profile: ProfileConfig, ohlcvs: OHLCVs) -> None:
        """Compute the profile's TRIX entry and exit signal columns."""
        symbol = profile.symbol
        timeframe = profile.timeframe
        signals = compute_signals(
            ohlcvs.column(symbol, timeframe, "close"),
            profile.trix_length,
            profile.signal_length,
            profile.signal_type,
            profile.trend_length,
            profile.trend_type,
        )
        ohlcvs.add_column(
            symbol, timeframe, profile.long_entry_column, signals.long_entry
        )
        ohlcvs.add_column(
            symbol, timeframe, profile.long_exit_column, signals.long_exit
        )
        ohlcvs.add_column(
            symbol, timeframe, profile.short_entry_column, signals.short_entry
        )
        ohlcvs.add_column(
            symbol, timeframe, profile.short_exit_column, signals.short_exit
        )

    def book_general_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: list[TimeFrame],
    ) -> None:
        self._closing_profile_ids = set()

    def book_profile_actions(
        self,
        profile: ProfileConfig,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
    ) -> None:
        """Emit the profile's entry or exit action for the current candle."""
        profile_snapshot = ProfileSnapshot.build(
            profile, ohlcvs, self._position_tracker
        )
        if profile_snapshot.has_no_price:
            return

        position = profile_snapshot.position
        if position is not None:
            if wants_exit(position, profile_snapshot):
                self._closing_profile_ids.add(profile.profile_id)
                book_exit(self.account, profile, position, bookkeeper)
            return

        side = wanted_entry_side(profile, profile_snapshot)
        if side is None:
            return
        book_entry(
            self.account,
            profile,
            side,
            profile_snapshot,
            account_snapshots.of(self.account),
            bookkeeper,
        )

    def book_final_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: list[TimeFrame],
    ) -> None:
        """Protect every open position, sparing the ones closing this candle."""
        book_position_protections(
            self.account,
            self.profiles,
            self._position_tracker,
            account_snapshots.of(self.account),
            self._closing_profile_ids,
            bookkeeper,
        )


def _validate_stop_loss_agreement(profiles: Sequence[ProfileConfig]) -> None:
    percentages_by_symbol: dict[Symbol, set[float | None]] = {}
    for profile in profiles:
        percentages_by_symbol.setdefault(profile.symbol, set()).add(
            profile.stop_loss_pct
        )
    for symbol, percentages in percentages_by_symbol.items():
        if len(percentages) > 1:
            raise BotCriticalError(
                f"Profiles on {symbol} set conflicting stop_loss_pct values. A symbol "
                f"has a single net position with one stop-loss, so stacked profiles "
                f"must all use the same stop_loss_pct (or all leave it unset)."
            )


def _margin_targets(profiles: Sequence[ProfileConfig]) -> dict[Symbol, MarginSettings]:
    targets: dict[Symbol, MarginSettings] = {}
    for profile in profiles:
        wanted = MarginSettings(
            leverage=profile.leverage, margin_mode=profile.margin_mode
        )
        declared = targets.setdefault(profile.symbol, wanted)
        if declared != wanted:
            raise BotCriticalError(
                f"Profiles on {profile.symbol} set conflicting leverage or "
                f"margin_mode values. A symbol trades at one leverage in one "
                f"margin mode, so stacked profiles must all use the same ones."
            )
    return targets


def _lookbacks_by_timeframe(profiles: Sequence[ProfileConfig]) -> dict[TimeFrame, int]:
    lookbacks: dict[TimeFrame, int] = {}
    for profile in profiles:
        needed = max(profile.trix_length, profile.signal_length, profile.trend_length)
        lookbacks[profile.timeframe] = max(lookbacks.get(profile.timeframe, 0), needed)
    return lookbacks


def _symbol_timeframes(profiles: Sequence[ProfileConfig]) -> list[SymbolTimeframe]:
    unique: dict[tuple[Symbol, TimeFrame], SymbolTimeframe] = {}
    for profile in profiles:
        key = (profile.symbol, profile.timeframe)
        unique.setdefault(
            key,
            SymbolTimeframe(symbol=profile.symbol, timeframe=profile.timeframe),
        )
    return list(unique.values())
