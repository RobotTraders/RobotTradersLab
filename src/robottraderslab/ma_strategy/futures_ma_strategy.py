import logging
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from robottraderslab._core import Symbol, TimeFrame
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.indicators import sma
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    BookKeeper,
    Candles,
    ChartLine,
    MarketType,
    OHLCVs,
    PositionSide,
    PositionTracker,
    Profile,
    ProfileStrategy,
    StrategyRequirements,
)
from robottraderslab.strategies.futures import (
    FuturesAccount,
    MarginMode,
    MarginSettings,
    PositionSnapshot,
    SizingRule,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ProfileConfig(Profile):
    """A profile sharing a symbol with another, on any timeframe, is told
    apart by its `tag`.
    """

    fast_ma_length: int
    slow_ma_length: int
    trend_ma_length: int | None = None
    long_only: bool = False
    short_only: bool = False
    leverage: float = 1.0
    margin_mode: MarginMode = MarginMode.ISOLATED
    sizing: SizingRule

    def __post_init__(self) -> None:
        """
        Raises:
            StrategyCriticalError: If both direction flags are set.
        """
        if self.long_only and self.short_only:
            raise StrategyCriticalError(
                "long_only and short_only cannot both be true; leave both out to "
                "trade both directions"
            )

    @property
    def is_long_short(self) -> bool:
        return self.trend_ma_length is not None

    @cached_property
    def signal_prefix(self) -> str:
        return f"{self.tag}_" if self.tag else ""


class FuturesMAStrategy(ProfileStrategy[ProfileConfig]):
    """Dual-mode MA strategy: plain crossover, or trend-filtered when a trend
    average is configured.
    """

    market_type: ClassVar[MarketType] = "futures"

    account: FuturesAccount
    _position_tracker: PositionTracker
    _pending_orders: dict[Symbol, PositionSide]

    def generate_profile_signals(self, profile: ProfileConfig, ohlcvs: OHLCVs) -> None:
        """Add the profile's MA crossover signals, trend-filtered in long/short mode."""
        if profile.is_long_short:
            _add_long_short_signals(ohlcvs, profile)
        else:
            _add_crossover_signals(ohlcvs, profile)

    def book_general_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: list[TimeFrame],
    ) -> None:
        self._pending_orders = {}

    def book_profile_actions(
        self,
        profile: ProfileConfig,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
    ) -> None:
        """Book the profile's entry or exit for the current candle."""
        account_snapshot = account_snapshots.of(self.account)
        close_value = ohlcvs.current(profile.symbol, profile.timeframe, "close")
        if math.isnan(close_value):
            return

        if profile.is_long_short:
            self._execute_long_short(
                ohlcvs,
                account_snapshot,
                close_value,
                profile,
                bookkeeper,
                self._pending_orders,
            )
        else:
            self._execute_crossover(
                ohlcvs,
                account_snapshot,
                close_value,
                profile,
                bookkeeper,
                self._pending_orders,
            )

    async def setup(self, requirements: StrategyRequirements) -> None:
        """Declare the data, account state and tracking the profiles need.

        Raises:
            StrategyCriticalError: If two profiles on the same symbol
                resolve to the same tag.
        """
        _reject_colliding_tags(self.profiles)
        for profile in self.profiles:
            lookback = profile.slow_ma_length
            if profile.trend_ma_length is not None:
                lookback = max(lookback, profile.trend_ma_length)
            requirements.ohlcv.add(profile.symbol, profile.timeframe, lookback)
        requirements.account.add(
            self.account,
            symbols=[profile.symbol for profile in self.profiles],
            positions=True,
            balances=True,
            margin_targets=_margin_targets(self.profiles),
            sizing=[profile.sizing for profile in self.profiles],
        )
        self._position_tracker = requirements.tracker.add(
            account=self.account,
            symbols={profile.profile_id: profile.symbol for profile in self.profiles},
            tags={profile.profile_id: profile.order_tag for profile in self.profiles},
        )

    def _execute_crossover(
        self,
        ohlcvs: OHLCVs,
        account_snapshot: AccountSnapshot,
        current_price: float,
        profile: ProfileConfig,
        bookkeeper: BookKeeper,
        pending_orders: dict[Symbol, PositionSide],
    ) -> None:
        p = profile.signal_prefix
        buy_signal = ohlcvs.signal(profile.symbol, profile.timeframe, f"{p}buy")
        sell_signal = ohlcvs.signal(profile.symbol, profile.timeframe, f"{p}sell")
        tracked_position = self._position_tracker.get(profile.profile_id)

        if tracked_position is not None:
            if sell_signal and tracked_position.side == PositionSide.LONG:
                action = (
                    self.account.long_exit(profile.symbol, tracked_position.quantity)
                    .tag(profile.order_tag)
                    .reason("ma crossover sell")
                    .build()
                )
                bookkeeper.add(action)
            elif buy_signal and tracked_position.side == PositionSide.SHORT:
                action = (
                    self.account.short_exit(profile.symbol, tracked_position.quantity)
                    .tag(profile.order_tag)
                    .reason("ma crossover buy")
                    .build()
                )
                bookkeeper.add(action)
        else:
            existing_position = account_snapshot.position(profile.symbol)
            pending_side = pending_orders.get(profile.symbol)

            if not profile.short_only and _should_enter_long(
                buy_signal, existing_position, pending_side
            ):
                pending_orders[profile.symbol] = PositionSide.LONG
                entry = self.account.long_entry(profile.symbol)
                action = (
                    entry.size(profile.sizing, current_price, account_snapshot)
                    .tag(profile.order_tag)
                    .reason("ma crossover buy")
                    .build()
                )
                bookkeeper.add(action)
            elif not profile.long_only and _should_enter_short(
                sell_signal, existing_position, pending_side
            ):
                pending_orders[profile.symbol] = PositionSide.SHORT
                entry = self.account.short_entry(profile.symbol)
                action = (
                    entry.size(profile.sizing, current_price, account_snapshot)
                    .tag(profile.order_tag)
                    .reason("ma crossover sell")
                    .build()
                )
                bookkeeper.add(action)

    def _execute_long_short(
        self,
        ohlcvs: OHLCVs,
        account_snapshot: AccountSnapshot,
        current_price: float,
        profile: ProfileConfig,
        bookkeeper: BookKeeper,
        pending_orders: dict[Symbol, PositionSide],
    ) -> None:
        p = profile.signal_prefix
        long_entry = ohlcvs.signal(profile.symbol, profile.timeframe, f"{p}long_entry")
        long_exit = ohlcvs.signal(profile.symbol, profile.timeframe, f"{p}long_exit")
        short_entry = ohlcvs.signal(
            profile.symbol, profile.timeframe, f"{p}short_entry"
        )
        short_exit = ohlcvs.signal(profile.symbol, profile.timeframe, f"{p}short_exit")
        tracked_position = self._position_tracker.get(profile.profile_id)

        if tracked_position is not None:
            if long_exit and tracked_position.side == PositionSide.LONG:
                action = (
                    self.account.long_exit(profile.symbol, tracked_position.quantity)
                    .tag(profile.order_tag)
                    .reason("trend long exit")
                    .build()
                )
                bookkeeper.add(action)
            elif short_exit and tracked_position.side == PositionSide.SHORT:
                action = (
                    self.account.short_exit(profile.symbol, tracked_position.quantity)
                    .tag(profile.order_tag)
                    .reason("trend short exit")
                    .build()
                )
                bookkeeper.add(action)
        else:
            existing_position = account_snapshot.position(profile.symbol)
            pending_side = pending_orders.get(profile.symbol)

            if not profile.short_only and _should_enter_long(
                long_entry, existing_position, pending_side
            ):
                pending_orders[profile.symbol] = PositionSide.LONG
                entry = self.account.long_entry(profile.symbol)
                action = (
                    entry.size(profile.sizing, current_price, account_snapshot)
                    .tag(profile.order_tag)
                    .reason("trend long entry")
                    .build()
                )
                bookkeeper.add(action)
            elif not profile.long_only and _should_enter_short(
                short_entry, existing_position, pending_side
            ):
                pending_orders[profile.symbol] = PositionSide.SHORT
                entry = self.account.short_entry(profile.symbol)
                action = (
                    entry.size(profile.sizing, current_price, account_snapshot)
                    .tag(profile.order_tag)
                    .reason("trend short entry")
                    .build()
                )
                bookkeeper.add(action)


def get_lightweight_chart_indicators(
    candles: Candles,
    indicator_params: dict[str, Any],
) -> list[ChartLine]:
    """Draw the averages this strategy crosses beside the candles.

    Args:
        indicator_params: `fast_ma_length` and `slow_ma_length`, plus an
            optional `trend_ma_length` drawn only when configured.
    """
    close = candles.close
    fast = indicator_params.get("fast_ma_length", 10)
    slow = indicator_params.get("slow_ma_length", 30)
    lines = [
        ChartLine(name=f"SMA {fast}", values=sma(close, fast), colour="blue"),
        ChartLine(name=f"SMA {slow}", values=sma(close, slow), colour="red"),
    ]

    trend = indicator_params.get("trend_ma_length")
    if trend is not None:
        lines.append(
            ChartLine(name=f"SMA {trend}", values=sma(close, trend), colour="green")
        )
    return lines


def _reject_colliding_tags(profiles: Iterable[ProfileConfig]) -> None:
    counts_by_symbol_tag: Counter[tuple[Symbol, str]] = Counter(
        (profile.symbol, profile.order_tag) for profile in profiles
    )
    colliding = {key: count for key, count in counts_by_symbol_tag.items() if count > 1}
    if colliding:
        detail = "; ".join(
            f"`{tag}` on {symbol}, declared by {count} profiles"
            for (symbol, tag), count in colliding.items()
        )
        raise StrategyCriticalError(f"profiles resolve to the same tag: {detail}")


def _margin_targets(profiles: Iterable[ProfileConfig]) -> dict[Symbol, MarginSettings]:
    """Profiles stacked on a symbol trade it at the first one's settings."""
    targets: dict[Symbol, MarginSettings] = {}
    for profile in profiles:
        targets.setdefault(
            profile.symbol,
            MarginSettings(leverage=profile.leverage, margin_mode=profile.margin_mode),
        )
    return targets


def _add_crossover_signals(ohlcvs: OHLCVs, profile: ProfileConfig) -> None:
    symbol = profile.symbol
    p = profile.signal_prefix
    close = ohlcvs.column(symbol, profile.timeframe, "close")

    fast_sma = sma(close, profile.fast_ma_length)
    slow_sma = sma(close, profile.slow_ma_length)

    prev_fast = _previous(fast_sma)
    prev_slow = _previous(slow_sma)

    ohlcvs.add_column(symbol, profile.timeframe, f"{p}fast_sma", fast_sma)
    ohlcvs.add_column(symbol, profile.timeframe, f"{p}slow_sma", slow_sma)
    ohlcvs.add_column(
        symbol,
        profile.timeframe,
        f"{p}buy",
        (fast_sma > slow_sma) & (prev_fast <= prev_slow),
    )
    ohlcvs.add_column(
        symbol,
        profile.timeframe,
        f"{p}sell",
        (fast_sma < slow_sma) & (prev_fast >= prev_slow),
    )


def _add_long_short_signals(ohlcvs: OHLCVs, profile: ProfileConfig) -> None:
    assert profile.trend_ma_length is not None
    symbol = profile.symbol
    p = profile.signal_prefix
    close = ohlcvs.column(symbol, profile.timeframe, "close")

    fast_sma = sma(close, profile.fast_ma_length)
    slow_sma = sma(close, profile.slow_ma_length)
    trend_sma = sma(close, profile.trend_ma_length)

    prev_fast = _previous(fast_sma)
    prev_slow = _previous(slow_sma)

    fast_cross_above_slow = (fast_sma > slow_sma) & (prev_fast <= prev_slow)
    fast_cross_below_slow = (fast_sma < slow_sma) & (prev_fast >= prev_slow)

    ohlcvs.add_column(symbol, profile.timeframe, f"{p}fast_sma", fast_sma)
    ohlcvs.add_column(symbol, profile.timeframe, f"{p}slow_sma", slow_sma)
    ohlcvs.add_column(symbol, profile.timeframe, f"{p}trend_sma", trend_sma)
    ohlcvs.add_column(
        symbol,
        profile.timeframe,
        f"{p}long_entry",
        fast_cross_above_slow & (close > trend_sma),
    )
    ohlcvs.add_column(symbol, profile.timeframe, f"{p}long_exit", fast_cross_below_slow)
    ohlcvs.add_column(
        symbol,
        profile.timeframe,
        f"{p}short_entry",
        fast_cross_below_slow & (close < trend_sma),
    )
    ohlcvs.add_column(
        symbol, profile.timeframe, f"{p}short_exit", fast_cross_above_slow
    )


def _previous(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    shifted = np.roll(values, 1)
    shifted[0] = np.nan
    return shifted


def _should_enter_long(
    entry_signal: bool,
    existing_position: PositionSnapshot | None,
    pending_side: PositionSide | None,
) -> bool:
    if not entry_signal:
        return False
    if existing_position and existing_position.side == PositionSide.SHORT:
        return False
    if pending_side == PositionSide.SHORT:
        return False
    return True


def _should_enter_short(
    entry_signal: bool,
    existing_position: PositionSnapshot | None,
    pending_side: PositionSide | None,
) -> bool:
    if not entry_signal:
        return False
    if existing_position and existing_position.side == PositionSide.LONG:
        return False
    if pending_side == PositionSide.LONG:
        return False
    return True
