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

from robottraderslab import Symbol, TimeFrame
from robottraderslab.exceptions import BotCriticalError, ProfileConfigError
from robottraderslab.indicators import crossover, crossunder, sma
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
    TrackedPosition,
)
from robottraderslab.strategies.futures import (
    FuturesAccount,
    FuturesOrderBuilder,
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

    A section setting both direction flags raises `ProfileConfigError`, which
    takes its symbol out of a live run while the other symbols trade. An edit
    of the configuration file that does this runs the bot's last working copy of
    it instead, with that symbol in exit-only mode.
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
            ProfileConfigError: If both direction flags are set.
        """
        if self.long_only and self.short_only:
            raise ProfileConfigError(
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

        self._execute(
            ohlcvs,
            account_snapshot,
            close_value,
            profile,
            _TREND_MODE if profile.is_long_short else _CROSSOVER_MODE,
            bookkeeper,
            self._pending_orders,
        )

    async def setup(self, requirements: StrategyRequirements) -> None:
        """Declare the data, account state and tracking the profiles need.

        Raises:
            BotCriticalError: If two profiles on the same symbol
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

    def _book_entry(
        self,
        ohlcvs: OHLCVs,
        profile: ProfileConfig,
        account_snapshot: AccountSnapshot,
        current_price: float,
        signal_mode: "_SignalMode",
        bookkeeper: BookKeeper,
        pending_orders: dict[Symbol, PositionSide],
    ) -> None:
        buy = _signal(ohlcvs, profile, signal_mode.entry_columns[PositionSide.LONG])
        sell = _signal(ohlcvs, profile, signal_mode.entry_columns[PositionSide.SHORT])
        existing_position = account_snapshot.position(profile.symbol)
        pending_side = pending_orders.get(profile.symbol)
        if not profile.short_only and _should_enter_long(
            buy, existing_position, pending_side
        ):
            position_side = PositionSide.LONG
            entry = self.account.long_entry(profile.symbol)
        elif not profile.long_only and _should_enter_short(
            sell, existing_position, pending_side
        ):
            position_side = PositionSide.SHORT
            entry = self.account.short_entry(profile.symbol)
        else:
            return
        pending_orders[profile.symbol] = position_side
        _book(
            bookkeeper,
            entry.size(profile.sizing, current_price, account_snapshot),
            profile,
            signal_mode.entry_reasons[position_side],
        )

    def _book_exit(
        self,
        ohlcvs: OHLCVs,
        profile: ProfileConfig,
        tracked_position: TrackedPosition,
        signal_mode: "_SignalMode",
        bookkeeper: BookKeeper,
    ) -> None:
        position_side = tracked_position.side
        signal_column = (
            "long_exit" if position_side == PositionSide.LONG else "short_exit"
        )
        if not _signal(ohlcvs, profile, signal_column):
            return
        exit_order = (
            self.account.long_exit(profile.symbol, tracked_position.quantity)
            if position_side == PositionSide.LONG
            else self.account.short_exit(profile.symbol, tracked_position.quantity)
        )
        _book(bookkeeper, exit_order, profile, signal_mode.exit_reasons[position_side])

    def _execute(
        self,
        ohlcvs: OHLCVs,
        account_snapshot: AccountSnapshot,
        current_price: float,
        profile: ProfileConfig,
        signal_mode: "_SignalMode",
        bookkeeper: BookKeeper,
        pending_orders: dict[Symbol, PositionSide],
    ) -> None:
        tracked_position = self._position_tracker.get(profile.profile_id)
        if tracked_position is not None:
            self._book_exit(ohlcvs, profile, tracked_position, signal_mode, bookkeeper)
            return
        self._book_entry(
            ohlcvs,
            profile,
            account_snapshot,
            current_price,
            signal_mode,
            bookkeeper,
            pending_orders,
        )


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
        raise BotCriticalError(f"profiles resolve to the same tag: {detail}")


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

    ohlcvs.add_column(symbol, profile.timeframe, f"{p}fast_sma", fast_sma)
    ohlcvs.add_column(symbol, profile.timeframe, f"{p}slow_sma", slow_sma)
    ohlcvs.add_column(
        symbol,
        profile.timeframe,
        f"{p}buy",
        crossover(fast_sma, slow_sma),
    )
    ohlcvs.add_column(
        symbol,
        profile.timeframe,
        f"{p}sell",
        crossunder(fast_sma, slow_sma),
    )
    _add_exit_states(ohlcvs, profile, fast_sma, slow_sma)


def _add_long_short_signals(ohlcvs: OHLCVs, profile: ProfileConfig) -> None:
    assert profile.trend_ma_length is not None
    symbol = profile.symbol
    p = profile.signal_prefix
    close = ohlcvs.column(symbol, profile.timeframe, "close")

    fast_sma = sma(close, profile.fast_ma_length)
    slow_sma = sma(close, profile.slow_ma_length)
    trend_sma = sma(close, profile.trend_ma_length)

    fast_cross_above_slow = crossover(fast_sma, slow_sma)
    fast_cross_below_slow = crossunder(fast_sma, slow_sma)

    ohlcvs.add_column(symbol, profile.timeframe, f"{p}fast_sma", fast_sma)
    ohlcvs.add_column(symbol, profile.timeframe, f"{p}slow_sma", slow_sma)
    ohlcvs.add_column(symbol, profile.timeframe, f"{p}trend_sma", trend_sma)
    ohlcvs.add_column(
        symbol,
        profile.timeframe,
        f"{p}long_entry",
        fast_cross_above_slow & (close > trend_sma),
    )
    ohlcvs.add_column(
        symbol,
        profile.timeframe,
        f"{p}short_entry",
        fast_cross_below_slow & (close < trend_sma),
    )
    _add_exit_states(ohlcvs, profile, fast_sma, slow_sma)


def _add_exit_states(
    ohlcvs: OHLCVs,
    profile: ProfileConfig,
    fast_sma: npt.NDArray[np.float64],
    slow_sma: npt.NDArray[np.float64],
) -> None:
    p = profile.signal_prefix
    ohlcvs.add_column(
        profile.symbol, profile.timeframe, f"{p}long_exit", fast_sma < slow_sma
    )
    ohlcvs.add_column(
        profile.symbol, profile.timeframe, f"{p}short_exit", fast_sma > slow_sma
    )


def _signal(ohlcvs: OHLCVs, profile: ProfileConfig, signal_column: str) -> bool:
    return ohlcvs.signal(
        profile.symbol, profile.timeframe, f"{profile.signal_prefix}{signal_column}"
    )


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


def _book(
    bookkeeper: BookKeeper,
    order_builder: FuturesOrderBuilder,
    profile: ProfileConfig,
    reason: str,
) -> None:
    bookkeeper.add(order_builder.tag(profile.order_tag).reason(reason).build())


@dataclass(frozen=True)
class _SignalMode:
    """The columns a profile enters on and the reasons its orders carry."""

    entry_columns: dict[PositionSide, str]
    entry_reasons: dict[PositionSide, str]
    exit_reasons: dict[PositionSide, str]


_CROSSOVER_MODE = _SignalMode(
    entry_columns={PositionSide.LONG: "buy", PositionSide.SHORT: "sell"},
    entry_reasons={
        PositionSide.LONG: "ma crossover buy",
        PositionSide.SHORT: "ma crossover sell",
    },
    exit_reasons={
        PositionSide.LONG: "ma crossover sell",
        PositionSide.SHORT: "ma crossover buy",
    },
)
_TREND_MODE = _SignalMode(
    entry_columns={PositionSide.LONG: "long_entry", PositionSide.SHORT: "short_entry"},
    entry_reasons={
        PositionSide.LONG: "trend long entry",
        PositionSide.SHORT: "trend short entry",
    },
    exit_reasons={
        PositionSide.LONG: "trend long exit",
        PositionSide.SHORT: "trend short exit",
    },
)
