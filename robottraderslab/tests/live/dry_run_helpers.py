import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, NamedTuple
from unittest.mock import Mock

import pytest
from live_cycle_helpers import BTC, ETH, logged_at_or_above

from robottraderslab import Symbol, TimeFrame
from robottraderslab._internal import DryRunFailure
from robottraderslab.exceptions import (
    BotCriticalError,
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ExchangeTransientError,
    MissingOhlcvDataError,
    ProfileConfigError,
    ProfileError,
)
from robottraderslab.exchanges import MarginMode, MarginSettings
from robottraderslab.futures import FuturesAccount
from robottraderslab.live.livebot import LiveBot
from robottraderslab.strategies import (
    Profile,
    ProfileStrategy,
    StrategyProtocol,
    TradingMode,
    TradingSystem,
)

MESSAGE = "boom"

_ERRORS: dict[str, Callable[[], BaseException]] = {
    "ValueError": lambda: ValueError(MESSAGE),
    "ProfileError": lambda: ProfileError(MESSAGE),
    "BotCriticalError": lambda: BotCriticalError(MESSAGE),
    "ExchangeCriticalError": lambda: ExchangeCriticalError(MESSAGE),
    "ExchangeRecoverableError": lambda: ExchangeRecoverableError(MESSAGE),
    "ExchangeTransientError": lambda: ExchangeTransientError(MESSAGE),
    "MissingOhlcvDataError": lambda: MissingOhlcvDataError(BTC, "1d"),
}

_WRITES = (
    "cancel_order_by_id",
    "cancel_orders_for_symbol",
    "modify_orders",
    "place_limit_order",
    "place_market_order",
    "place_orders",
    "place_trailing_stop_order",
    "set_leverage",
    "set_margin_mode",
    "update_position_stop_loss",
    "update_position_take_profit",
    "update_position_trailing_stop",
    "update_position_trailing_stop_ratio",
)

HELD_SETTINGS = MarginSettings(leverage=1.0, margin_mode=MarginMode.ISOLATED)
CANCELLED_ORDER = "resting-1"


class BookedCall(NamedTuple):
    """One call of a profile's booking slot."""

    symbol: Symbol
    timeframe: TimeFrame
    quantity: float


@dataclass(frozen=True, kw_only=True)
class _ProbeProfile(Profile):
    """A profile whose `fail` says how a slot fails, as `<phase>:<error>`.

    `quantity` and `leverage` size its entry, `margin_mode` is the mode it
    declares, `closes` books a close before the entry, `sends_everything` books a
    cancel, a leverage and a margin mode beside it, and `refused`, `raises`, `critical` and
    `venue_critical` make its section fail to build, the last two with a
    `BotCriticalError` and an `ExchangeCriticalError`.
    """

    quantity: float = 1.0
    leverage: float = 1.0
    margin_mode: str = "isolated"
    refused: bool = False
    raises: bool = False
    critical: bool = False
    venue_critical: bool = False
    closes: bool = False
    sends_everything: bool = False
    fail: str = ""

    def __post_init__(self) -> None:
        if self.refused:
            raise ProfileConfigError("this section is refused")
        if self.raises:
            raise RuntimeError("this section cannot be built")
        if self.critical:
            raise BotCriticalError("this section stops the bot")
        if self.venue_critical:
            raise ExchangeCriticalError("this section meets a venue refusal")


class ProbeStrategy(ProfileStrategy[_ProbeProfile]):
    """A strategy whose profiles say how a slot fails, a general one included.

    Each profile declares the leverage it wants and books an entry sized by the
    leverage its snapshot holds, so the quantity booked tells what leverage the
    strategy saw. `booked` and `signalled` list the profile slots it ran and
    `phases` the general slots and `setup`, in order.
    """

    market_type: ClassVar[str] = "futures"

    def __init__(self, **arguments: Any) -> None:
        super().__init__(**arguments)
        self.booked: list[BookedCall] = []
        self.phases: list[str] = []
        self.signalled: list[Symbol] = []

    def book_final_actions(self, *arguments: Any) -> None:
        self._enter_phase("final_booking")

    def book_general_actions(self, *arguments: Any) -> None:
        self._enter_phase("general_booking")

    def book_profile_actions(
        self,
        profile: _ProbeProfile,
        ohlcvs: Any,
        account_snapshots: Any,
        timestamp: Any,
        bookkeeper: Any,
    ) -> None:
        held = account_snapshots.of(self.account).margin_settings(profile.symbol)
        quantity = profile.quantity * held.leverage
        self.booked.append(BookedCall(profile.symbol, profile.timeframe, quantity))
        _raise_if_phase(profile.fail, "booking")
        if profile.closes:
            bookkeeper.add(self.account.close_position(profile.symbol))
        bookkeeper.add(self.account.long_entry(profile.symbol, quantity))
        if profile.sends_everything:
            bookkeeper.add(self.account.cancel_order(profile.symbol, CANCELLED_ORDER))
            bookkeeper.add(self.account.set_leverage(profile.symbol, 7.0))
            bookkeeper.add(
                self.account.set_margin_mode(profile.symbol, MarginMode.CROSS)
            )

    def generate_general_signals(self, ohlcvs: Any) -> None:
        self._enter_phase("general_signals")

    def generate_profile_signals(self, profile: _ProbeProfile, ohlcvs: Any) -> None:
        self.signalled.append(profile.symbol)
        _raise_if_phase(profile.fail, "signals")

    async def setup(self, requirements: Any) -> None:
        self._enter_phase("setup")
        targets = {
            profile.symbol: MarginSettings(
                leverage=profile.leverage, margin_mode=MarginMode(profile.margin_mode)
            )
            for profile in self.profiles
        }
        for profile in self.profiles:
            requirements.ohlcv.add(profile.symbol, profile.timeframe)
        requirements.account.add(
            self.account,
            symbols=list(targets),
            positions=True,
            open_orders=True,
            margin_settings=True,
            margin_targets=targets,
        )

    def _enter_phase(self, phase: str) -> None:
        self.phases.append(phase)
        for profile in self.profiles:
            _raise_if_phase(profile.fail, phase)


class PlainStrategy(StrategyProtocol):
    """A strategy that is no profile strategy, declaring three timeframes."""

    market_type = "futures"

    def __init__(self, account: FuturesAccount) -> None:
        self._account = account
        self.signal_calls = 0
        self.booking_timeframes: list[list[TimeFrame]] = []

    def book_trading_actions(
        self,
        ohlcvs: Any,
        account_snapshots: Any,
        timestamp: datetime,
        bookkeeper: Any,
        triggered_timeframes: Iterable[TimeFrame],
    ) -> None:
        self.booking_timeframes.append(list(triggered_timeframes))

    def generate_trading_signals(self, ohlcvs: Any) -> None:
        self.signal_calls += 1

    async def setup(self, requirements: Any) -> None:
        for symbol in (BTC, ETH):
            for timeframe in ("1d", "1w", "1h"):
                requirements.ohlcv.add(symbol, timeframe)
        requirements.account.add(
            self._account, symbols=[BTC, ETH], positions=True, open_orders=True
        )


def lines_a_failure_must_not_log(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The WARNING lines and the probe-message lines the scenarios must not log.

    The log's lines at WARNING or above, and its INFO lines carrying the
    message the probe errors raise.
    """
    return logged_at_or_above(caplog, logging.WARNING) + [
        line for line in logged_at_or_above(caplog, logging.INFO) if MESSAGE in line
    ]


def section(
    symbol: Symbol, timeframe: TimeFrame = "1d", **fields: Any
) -> dict[str, Any]:
    """A profile section of a probe strategy."""
    return {"symbol": str(symbol), "timeframe": timeframe, **fields}


def probe(account: FuturesAccount, *sections: dict[str, Any]) -> ProbeStrategy:
    """A probe strategy of the sections."""
    return ProbeStrategy(
        account=account,
        trading_system=TradingSystem(trading_mode=TradingMode.LIVE),
        config_dir=None,
        profiles=sections,
    )


def btc_probe(account: FuturesAccount, **btc_fields: Any) -> ProbeStrategy:
    """A probe strategy of one BTC profile, with the fields."""
    return probe(account, section(BTC, **btc_fields))


def btc_and_eth_probe(account: FuturesAccount, **btc_fields: Any) -> ProbeStrategy:
    """A probe strategy of a BTC profile with the fields and a plain ETH one."""
    return probe(account, section(BTC, **btc_fields), section(ETH))


async def dry_run(
    strategy: StrategyProtocol, provider: Mock, **options: Any
) -> list[DryRunFailure]:
    """Dry run one cycle of the strategy and return what it met."""
    livebot = LiveBot(strategy, provider, dry_run=True, **options)
    await livebot.run()
    return livebot.failures


async def dry_run_at_debug(
    caplog: pytest.LogCaptureFixture, strategy: StrategyProtocol, provider: Mock
) -> list[DryRunFailure]:
    """Dry run one cycle, capturing every level from an empty log."""
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        return await dry_run(strategy, provider)


async def dry_run_probe(
    account: FuturesAccount,
    sections: Iterable[dict[str, Any]],
    provider: Mock,
    **options: Any,
) -> list[DryRunFailure]:
    """Dry run one cycle of a probe strategy of the sections."""
    return await dry_run(probe(account, *sections), provider, **options)


def sent(venue: Mock) -> list[str]:
    """The names of the write calls the venue received.

    Args:
        venue: A mock of the exchange protocol, which has each of the thirteen
            write methods this module lists.
    """
    return [name for name in _WRITES if getattr(venue, name).await_count]


def _raise_if_phase(spec: str, phase: str) -> None:
    """Raise the error `spec` names when it is the phase's, `<phase>:<error>`.

    The error is one of the names this module defines, `ValueError` when the
    spec names none; any other name raises KeyError.
    """
    where, _, error = spec.partition(":")
    if where == phase:
        raise _ERRORS[error or "ValueError"]()
