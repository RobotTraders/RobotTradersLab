import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal
from unittest.mock import Mock

import pytest
from readouts import outcomes

from robottraderslab import Symbol, TimeFrame
from robottraderslab._internal import (
    DryRunFailure,
    collect_profile_failures,
    collecting_refusals,
    log_strategy_slots,
    sitting_out_symbols,
)
from robottraderslab.exceptions import (
    BotCriticalError,
    ExchangeCriticalError,
    ExchangeRecoverableError,
    MissingOhlcvDataError,
    ProfileConfigError,
    ProfileError,
)
from robottraderslab.strategies import (
    BookKeeper,
    OHLCVs,
    Profile,
    ProfileStrategy,
    StrategyProtocol,
    TradingMode,
    TradingSystem,
)
from robottraderslab.strategies.futures import (
    AvailableBalanceRatio,
    EquityRatio,
    Margin,
    Notional,
    Quantity,
    RiskRatio,
    SizingRule,
    TotalBalanceRatio,
)

TIMESTAMP = datetime(2024, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _Profile:
    profile_id: str
    timeframe: TimeFrame


class _Pace(StrEnum):
    FAST = "fast"
    SLOW = "slow"


@dataclass(frozen=True, kw_only=True)
class _TypedProfile(Profile):
    length: int = 20
    pace: _Pace = _Pace.FAST

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise BotCriticalError(f"length must be positive, got {self.length}")


@dataclass(frozen=True, kw_only=True)
class _CrossingProfile(Profile):
    fast: int
    slow: int

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ProfileConfigError(
                f"fast ({self.fast}) must be below slow ({self.slow})"
            )


class _CrossingStrategy(ProfileStrategy[_CrossingProfile]):
    market_type: ClassVar[str] = "futures"


_RAISED_BY_POST_INIT = {
    "ValueError": ValueError,
    "AssertionError": AssertionError,
    "KeyError": KeyError,
    "ZeroDivisionError": ZeroDivisionError,
    "TypeError": TypeError,
    "BotCriticalError": BotCriticalError,
}


@dataclass(frozen=True, kw_only=True)
class _ProbeProfile(Profile):
    sizing: SizingRule
    raises: str = ""

    def __post_init__(self) -> None:
        if self.raises:
            raise _RAISED_BY_POST_INIT[self.raises]("probe refusal")


class _ProbeStrategy(ProfileStrategy[_ProbeProfile]):
    market_type: ClassVar[str] = "futures"


class _TypedStrategy(ProfileStrategy[_TypedProfile]):
    market_type: ClassVar[str] = "futures"
    entry_order: Literal["trigger", "limit"] = "trigger"
    lookback: int


@dataclass(frozen=True)
class _NoTimeframe:
    profile_id: str


@dataclass(frozen=True, slots=True)
class _MyRiskRatio(SizingRule):
    ratio: float


@dataclass(frozen=True, slots=True)
class _MyKelly(SizingRule):
    lookback: int


@dataclass(frozen=True, kw_only=True)
class _SizedProfile(Profile):
    sizing: SizingRule


class _SizedStrategy(ProfileStrategy[_SizedProfile]):
    sizing_rules = {"my_risk_ratio": _MyRiskRatio, "my_kelly": _MyKelly}


class _StrategySizedOnce(ProfileStrategy[_TypedProfile]):
    sizing: SizingRule


class _NoTimeframeStrategy(ProfileStrategy[_NoTimeframe]):
    pass


class _RecordingStrategy(ProfileStrategy[_Profile]):
    def __init__(
        self,
        *,
        failing: Exception | None = None,
        failing_profile_id: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._failing = failing
        self._failing_profile_id = failing_profile_id
        self.generated: list[str] = []
        self.booked: list[str] = []

    def generate_general_signals(self, ohlcvs: OHLCVs) -> None:
        self.generated.append("general")

    def generate_profile_signals(self, profile: _Profile, ohlcvs: OHLCVs) -> None:
        _raise_or_record(
            self._failing,
            profile.profile_id == self._failing_profile_id,
            self.generated,
            profile.profile_id,
        )

    def book_general_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: Any,
        timestamp: datetime,
        bookkeeper: Any,
        triggered_timeframes: list[TimeFrame],
    ) -> None:
        self.booked.append("general")

    def book_profile_actions(
        self,
        profile: _Profile,
        ohlcvs: OHLCVs,
        account_snapshots: Any,
        timestamp: datetime,
        bookkeeper: Any,
    ) -> None:
        _raise_or_record(
            self._failing,
            profile.profile_id == self._failing_profile_id,
            self.booked,
            profile.profile_id,
        )

    def book_final_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: Any,
        timestamp: datetime,
        bookkeeper: Any,
        triggered_timeframes: list[TimeFrame],
    ) -> None:
        self.booked.append("final")


class _SignalFailingStrategy(_RecordingStrategy):
    def __init__(
        self, *, signal_error: Exception, failing_cycles: int = 1, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._signal_error = signal_error
        self._failing_cycles = failing_cycles

    def generate_profile_signals(self, profile: _Profile, ohlcvs: OHLCVs) -> None:
        if profile.profile_id == "alpha" and self._failing_cycles > 0:
            self._failing_cycles -= 1
            raise self._signal_error
        super().generate_profile_signals(profile, ohlcvs)


class _SymbolStrategy(ProfileStrategy[_CrossingProfile]):
    market_type: ClassVar[str] = "futures"

    def __init__(
        self,
        *,
        failing_symbol: Symbol,
        signal_error: Exception | None = None,
        booking_error: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._signal_error = signal_error
        self._booking_error = booking_error
        self._failing_symbol = failing_symbol
        self._signalled: list[Symbol] = []
        self._booked: list[Symbol] = []

    @property
    def booked(self) -> list[Symbol]:
        return self._booked

    @property
    def signalled(self) -> list[Symbol]:
        return self._signalled

    def book_profile_actions(
        self,
        profile: _CrossingProfile,
        ohlcvs: OHLCVs,
        account_snapshots: Any,
        timestamp: datetime,
        bookkeeper: Any,
    ) -> None:
        _raise_or_record(
            self._booking_error,
            profile.symbol == self._failing_symbol,
            self._booked,
            profile.symbol,
        )

    def generate_profile_signals(
        self, profile: _CrossingProfile, ohlcvs: OHLCVs
    ) -> None:
        _raise_or_record(
            self._signal_error,
            profile.symbol == self._failing_symbol,
            self._signalled,
            profile.symbol,
        )


class _Unset(ProfileStrategy[_Profile]):
    def __init__(self) -> None:
        self._profiles = PROFILES
        self._trading_mode = TradingMode.BACKTEST


class _BareStrategy(ProfileStrategy[_Profile]):
    pass


class _ThreeSlotStrategy(ProfileStrategy[_Profile]):
    def generate_profile_signals(self, profile: _Profile, ohlcvs: OHLCVs) -> None:
        pass

    def book_general_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: Any,
        timestamp: datetime,
        bookkeeper: Any,
        triggered_timeframes: list[TimeFrame],
    ) -> None:
        pass

    def book_profile_actions(
        self,
        profile: _Profile,
        ohlcvs: OHLCVs,
        account_snapshots: Any,
        timestamp: datetime,
        bookkeeper: Any,
    ) -> None:
        pass


ACCOUNT = object()
CONFIG_DIR = Path("bot-config-dir")
PROFILES = [
    {"profile_id": "alpha", "timeframe": "1d"},
    {"profile_id": "beta", "timeframe": "1d"},
]
SECTION = {"symbol": "BTC/USDT:USDT", "timeframe": "4h"}
ACCEPTED_CROSSING = {**SECTION, "fast": 10, "slow": 30}
REFUSED_CROSSING = {
    "symbol": "ETH/USDT:USDT",
    "timeframe": "1h",
    "fast": 50,
    "slow": 30,
}


VALID_PROBE = {"symbol": "BTC/USDT:USDT", "timeframe": "1h", "quantity": 1}
REFUSED_PROBE_SECTION = {"symbol": "ETH/USDT:USDT", "timeframe": "1h", "quantity": 1}
UNBUILDABLE_SECTIONS = {
    "unknown key": ({"lenght": 30}, "lenght: unknown key"),
    "wrong type": ({"raises": 5}, "raises: Input should be a valid string"),
    "two sizing rules": ({"notional": 200}, "names several sizing rules"),
    "sizing value its rule refuses": (
        {"quantity": -1},
        "quantity: `quantity` must be greater than 0",
    ),
    "ValueError": ({"raises": "ValueError"}, "probe refusal"),
    "AssertionError": ({"raises": "AssertionError"}, "Assertion failed, probe refusal"),
    "KeyError": ({"raises": "KeyError"}, "KeyError: 'probe refusal'"),
    "ZeroDivisionError": (
        {"raises": "ZeroDivisionError"},
        "ZeroDivisionError: probe refusal",
    ),
    "TypeError": ({"raises": "TypeError"}, "TypeError: probe refusal"),
}
ACCEPTED_CROSSING_ON_SOL = {**ACCEPTED_CROSSING, "symbol": "SOL/USDT:USDT"}
ACCEPTED_CROSSING_ON_ETH = {**ACCEPTED_CROSSING, "symbol": "ETH/USDT:USDT"}
REFUSED_CROSSING_ON_BTC = {**REFUSED_CROSSING, "symbol": "BTC/USDT:USDT"}
BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")


def _create[T: ProfileStrategy[Any]](
    strategy_class: type[T],
    profiles: list[dict[str, Any]],
    trading_mode: TradingMode,
    **settings: Any,
) -> T:
    return strategy_class(
        account=ACCOUNT,
        trading_system=TradingSystem(trading_mode=trading_mode),
        config_dir=CONFIG_DIR,
        profiles=profiles,
        **settings,
    )


def _raise_or_record(
    error: Exception | None, fails: bool, record: list[Any], key: Any
) -> None:
    if error and fails:
        raise error
    record.append(key)


def _refused_on_btc_strategy() -> _CrossingStrategy:
    return _create(
        _CrossingStrategy,
        [ACCEPTED_CROSSING, REFUSED_CROSSING_ON_BTC, ACCEPTED_CROSSING_ON_SOL],
        TradingMode.LIVE,
    )


def _book(strategy: ProfileStrategy[Any], triggered: list[TimeFrame]) -> None:
    strategy.book_trading_actions(object(), object(), TIMESTAMP, object(), triggered)


class TestSignalGeneration:
    def test_slot_order(self):
        strategy = _create(_RecordingStrategy, PROFILES, TradingMode.BACKTEST)

        strategy.generate_trading_signals(object())

        assert strategy.generated == ["general", "alpha", "beta"]

    def test_a_failing_profile_in_live(self, caplog):
        strategy = _create(
            _RecordingStrategy,
            PROFILES,
            TradingMode.LIVE,
            failing=KeyError("boom"),
            failing_profile_id="alpha",
        )

        strategy.generate_trading_signals(object())

        assert strategy.generated == ["general", "beta"]
        assert "alpha" in caplog.text

    def test_a_failing_profile_in_backtest(self):
        strategy = _create(
            _RecordingStrategy,
            PROFILES,
            TradingMode.BACKTEST,
            failing=KeyError("boom"),
            failing_profile_id="alpha",
        )

        with pytest.raises(KeyError):
            strategy.generate_trading_signals(object())

    def test_a_critical_error_in_live(self):
        strategy = _create(
            _RecordingStrategy,
            PROFILES,
            TradingMode.LIVE,
            failing=ExchangeCriticalError("auth"),
            failing_profile_id="alpha",
        )

        with pytest.raises(ExchangeCriticalError):
            strategy.generate_trading_signals(object())


class TestBooking:
    def test_slot_order(self):
        strategy = _create(_RecordingStrategy, PROFILES, TradingMode.BACKTEST)

        _book(strategy, ["1d"])

        assert strategy.booked == ["general", "alpha", "beta", "final"]

    def test_a_profile_whose_timeframe_did_not_close(self):
        profiles = [
            {"profile_id": "alpha", "timeframe": "1d"},
            {"profile_id": "beta", "timeframe": "4h"},
        ]
        strategy = _create(_RecordingStrategy, profiles, TradingMode.BACKTEST)

        _book(strategy, ["4h"])

        assert strategy.booked == ["general", "beta", "final"]

    def test_a_failing_profile_in_live(self, caplog):
        strategy = _create(
            _RecordingStrategy,
            PROFILES,
            TradingMode.LIVE,
            failing=KeyError("boom"),
            failing_profile_id="alpha",
        )

        _book(strategy, ["1d"])

        assert strategy.booked == ["general", "beta", "final"]
        assert "alpha" in caplog.text

    def test_a_failing_profile_in_backtest(self):
        strategy = _create(
            _RecordingStrategy,
            PROFILES,
            TradingMode.BACKTEST,
            failing=KeyError("boom"),
            failing_profile_id="alpha",
        )

        with pytest.raises(KeyError):
            _book(strategy, ["1d"])


def _generate(strategy: ProfileStrategy[Any]) -> None:
    strategy.generate_trading_signals(object())


def _book_every_profile(strategy: ProfileStrategy[Any]) -> None:
    _book(strategy, ["1d"])


SLOT_RUNS = [(_generate, "signals"), (_book_every_profile, "booking")]


class TestSlotFailureRecords:
    @pytest.mark.parametrize(("run", "slot"), SLOT_RUNS)
    @pytest.mark.parametrize(
        ("error", "level", "traceback_kept"),
        [
            (KeyError("boom"), logging.ERROR, True),
            (ProfileError("stated on purpose"), logging.ERROR, False),
            (ProfileConfigError("refused on purpose"), logging.ERROR, False),
            (MissingOhlcvDataError("BTC/USDT:USDT", "1d"), logging.WARNING, False),
            (ExchangeRecoverableError("rejected"), logging.WARNING, False),
        ],
        ids=[
            "unexpected exception",
            "profile error",
            "profile refusal",
            "missing candles",
            "recoverable error",
        ],
    )
    def test_the_record_names_the_profile_the_slot_and_the_exception_class(
        self, caplog, run, slot, error, level, traceback_kept
    ):
        strategy = _create(
            _RecordingStrategy,
            PROFILES,
            TradingMode.LIVE,
            failing=error,
            failing_profile_id="alpha",
        )

        run(strategy)

        (record,) = caplog.records
        assert record.levelno == level
        assert record.getMessage() == (
            f"Profile alpha sat out this candle: the {slot} slot raised "
            f"{type(error).__name__}: {error}"
        )
        assert (record.exc_info is not None) is traceback_kept

    @pytest.mark.parametrize("run", [_generate, _book_every_profile])
    @pytest.mark.parametrize(
        "error", [ExchangeCriticalError("auth"), BotCriticalError("broken profile")]
    )
    def test_a_critical_error_stops_the_bot(self, run, error):
        strategy = _create(
            _RecordingStrategy,
            PROFILES,
            TradingMode.LIVE,
            failing=error,
            failing_profile_id="alpha",
        )

        with pytest.raises(type(error)):
            run(strategy)

    @pytest.mark.parametrize("run", [_generate, _book_every_profile])
    @pytest.mark.parametrize("error", [BotCriticalError("broken"), ProfileError("x")])
    def test_a_backtest_stops_on_whatever_a_slot_raises(self, run, error):
        strategy = _create(
            _RecordingStrategy,
            PROFILES,
            TradingMode.BACKTEST,
            failing=error,
            failing_profile_id="alpha",
        )

        with pytest.raises(type(error)):
            run(strategy)


class TestProfileWhoseSignalsFailed:
    @pytest.mark.parametrize(
        "error",
        [
            KeyError("boom"),
            MissingOhlcvDataError("BTC/USDT:USDT", "1d"),
        ],
        ids=["unexpected exception", "missing candles"],
    )
    def test_live_books_nothing_for_it_and_the_other_profile_books(self, error):
        strategy = _create(
            _SignalFailingStrategy, PROFILES, TradingMode.LIVE, signal_error=error
        )
        strategy.generate_trading_signals(object())

        _book(strategy, ["1d"])

        assert strategy.booked == ["general", "beta", "final"]

    def test_live_books_for_it_again_on_the_next_cycle(self):
        strategy = _create(
            _SignalFailingStrategy,
            PROFILES,
            TradingMode.LIVE,
            signal_error=KeyError("boom"),
        )
        strategy.generate_trading_signals(object())
        strategy.generate_trading_signals(object())

        _book(strategy, ["1d"])

        assert strategy.booked == ["general", "alpha", "beta", "final"]

    def test_a_backtest_calls_its_booking_slot(self):
        strategy = _create(
            _SignalFailingStrategy,
            PROFILES,
            TradingMode.BACKTEST,
            signal_error=KeyError("boom"),
        )
        with pytest.raises(KeyError):
            strategy.generate_trading_signals(object())

        _book(strategy, ["1d"])

        assert strategy.booked == ["general", "alpha", "beta", "final"]


class TestInitialisation:
    def test_the_account_and_config_dir_kept(self):
        strategy = _create(_BareStrategy, PROFILES, TradingMode.BACKTEST)

        assert strategy.account is ACCOUNT
        assert strategy.config_dir == CONFIG_DIR

    def test_a_subclass_of_a_profile_strategy(self):
        class _Child(_RecordingStrategy):
            pass

        strategy = _create(_Child, PROFILES, TradingMode.BACKTEST)

        strategy.generate_trading_signals(object())

        assert strategy.generated == ["general", "alpha", "beta"]


class TestProfileBuild:
    def test_a_section_built_by_its_field_types(self):
        section = {**SECTION, "pace": "slow", "length": "30", "tag": "r1"}

        strategy = _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)

        (profile,) = strategy.profiles
        assert profile.pace is _Pace.SLOW
        assert profile.length == 30
        assert profile.profile_id == "BTC/USDT:USDT@4h-r1"
        assert profile.order_tag == "4h-r1"
        assert str(profile.symbol) == "BTC/USDT:USDT"

    @pytest.mark.parametrize("tag", ["sl", "tp"])
    def test_a_tag_a_bracket_id_ends_with_is_refused_at_load(self, tag):
        section = {**SECTION, "tag": tag}

        with pytest.raises(
            BotCriticalError,
            match=rf"profile 1 \(.*\): tag: '{tag}' ends a bracket's client order id",
        ):
            _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)

    def test_defaults_when_a_section_leaves_a_field_out(self):
        strategy = _create(_TypedStrategy, [SECTION], TradingMode.BACKTEST, lookback=5)

        (profile,) = strategy.profiles
        assert profile.pace is _Pace.FAST
        assert profile.length == 20
        assert profile.tag == ""
        assert profile.profile_id == "BTC/USDT:USDT@4h"

    def test_a_value_the_field_type_refuses(self):
        section = {**SECTION, "pace": "slo"}

        with pytest.raises(
            BotCriticalError,
            match=r"profile 1 \(BTC/USDT:USDT 4h\): pace: Input should be 'fast' or 'slow'",
        ):
            _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)

    def test_a_key_the_profile_has_no_field_for(self):
        section = {**SECTION, "lenght": 30}

        with pytest.raises(
            BotCriticalError, match=r"profile 1 \(.*\): lenght: unknown key"
        ):
            _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)

    def test_a_required_field_left_out(self):
        with pytest.raises(
            BotCriticalError, match=r"profile 2 \(4h\): symbol: Field required"
        ):
            _create(
                _TypedStrategy,
                [SECTION, {"timeframe": "4h"}],
                TradingMode.BACKTEST,
                lookback=5,
            )

    def test_the_profiles_own_rule(self):
        section = {**SECTION, "length": 0}

        with pytest.raises(BotCriticalError, match=r"^length must be positive"):
            _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)

    def test_a_symbol_spelt_wrong(self):
        section = {**SECTION, "symbol": "BTCUSDT"}

        with pytest.raises(
            BotCriticalError, match="symbol: Invalid symbol format: BTCUSDT"
        ):
            _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)

    @pytest.mark.parametrize("tag", ["a@b", "a-b", "a/b", "a:b", "a b"])
    def test_a_tag_with_a_reserved_character(self, tag):
        section = {**SECTION, "tag": tag}

        with pytest.raises(
            BotCriticalError, match=f"tag: '{re.escape(tag)}' cannot contain"
        ):
            _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)

    def test_a_tag_over_eight_bytes(self):
        section = {**SECTION, "tag": "abcdefghi"}

        with pytest.raises(
            BotCriticalError,
            match=r"profile 1 \(.*\): tag: 'abcdefghi' is 9 bytes, over the 8-byte",
        ):
            _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)


class TestRefusedProfile:
    def test_a_backtest_stops_naming_the_profile_and_the_reason(self):
        with pytest.raises(
            ProfileConfigError,
            match=r"^_CrossingStrategy: profile 2 \(ETH/USDT:USDT 1h\): "
            r"fast \(50\) must be below slow \(30\)$",
        ):
            _create(
                _CrossingStrategy,
                [ACCEPTED_CROSSING, REFUSED_CROSSING],
                TradingMode.BACKTEST,
            )

    def test_live_leaves_the_refused_profile_out_and_keeps_the_others(self):
        strategy = _create(
            _CrossingStrategy,
            [REFUSED_CROSSING, ACCEPTED_CROSSING],
            TradingMode.LIVE,
        )

        assert [str(profile.symbol) for profile in strategy.profiles] == [
            "BTC/USDT:USDT"
        ]

    def test_live_names_the_refused_profile_in_one_error_line(self, caplog):
        with caplog.at_level(logging.INFO):
            _create(
                _CrossingStrategy,
                [REFUSED_CROSSING, ACCEPTED_CROSSING],
                TradingMode.LIVE,
            )

        assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
            (
                logging.ERROR,
                "_CrossingStrategy: profile 1 (ETH/USDT:USDT 1h): fast (50) must be "
                "below slow (30); this profile sits out the run",
            )
        ]

    def test_live_collects_each_refused_section_even_when_the_build_stops(self):
        with collecting_refusals() as refusals:
            with pytest.raises(BotCriticalError):
                _create(
                    _CrossingStrategy,
                    [REFUSED_CROSSING, REFUSED_CROSSING],
                    TradingMode.LIVE,
                )

        assert [str(failure.symbol) for failure in refusals] == ["ETH/USDT:USDT"] * 2
        assert [failure.recoverable for failure in refusals] == [False, False]

    def test_live_collects_nothing_outside_the_block(self):
        with collecting_refusals() as refusals:
            pass

        _create(
            _CrossingStrategy, [REFUSED_CROSSING, ACCEPTED_CROSSING], TradingMode.LIVE
        )

        assert refusals == []

    def test_live_with_every_profile_refused_stops_naming_the_strategy(self):
        with pytest.raises(
            BotCriticalError,
            match=r"^_CrossingStrategy: every profile was refused or left out with a "
            r"refused profile's symbol, so no profile is left to run; correct the "
            r"profile sections the ERROR lines above name",
        ):
            _create(_CrossingStrategy, [REFUSED_CROSSING], TradingMode.LIVE)

    @pytest.mark.parametrize(
        ("changes", "reason"),
        list(UNBUILDABLE_SECTIONS.values()),
        ids=list(UNBUILDABLE_SECTIONS),
    )
    def test_live_leaves_out_a_section_the_engine_cannot_build(
        self, changes, reason, caplog
    ):
        refused = {**REFUSED_PROBE_SECTION, **changes}

        strategy = _create(_ProbeStrategy, [VALID_PROBE, refused], TradingMode.LIVE)

        assert [str(profile.symbol) for profile in strategy.profiles] == [
            "BTC/USDT:USDT"
        ]
        assert sitting_out_symbols(strategy) == {Symbol.create("ETH/USDT:USDT")}
        (error_record,) = (r for r in caplog.records if r.levelno == logging.ERROR)
        assert error_record.getMessage().startswith(
            "_ProbeStrategy: profile 2 (ETH/USDT:USDT 1h): "
        )
        assert reason in error_record.getMessage()

    @pytest.mark.parametrize("trading_mode", [TradingMode.LIVE, TradingMode.BACKTEST])
    def test_a_profile_class_raising_a_bot_critical_error_stops_the_bot(
        self, trading_mode
    ):
        refused = {**REFUSED_PROBE_SECTION, "raises": "BotCriticalError"}

        with pytest.raises(BotCriticalError, match=r"^probe refusal$"):
            _create(_ProbeStrategy, [VALID_PROBE, refused], trading_mode)

    @pytest.mark.parametrize(
        ("raises", "reason"),
        [
            ("ValueError", "probe refusal"),
            ("AssertionError", "Assertion failed, probe refusal"),
        ],
    )
    def test_live_names_a_reason_that_has_no_key_alone(self, caplog, raises, reason):
        refused = {**REFUSED_PROBE_SECTION, "raises": raises}

        _create(_ProbeStrategy, [VALID_PROBE, refused], TradingMode.LIVE)

        (error_record,) = (r for r in caplog.records if r.levelno == logging.ERROR)
        assert error_record.getMessage() == (
            f"_ProbeStrategy: profile 2 (ETH/USDT:USDT 1h): {reason}; "
            "this profile sits out the run"
        )

    def test_a_backtest_names_a_reason_that_has_no_key_alone(self):
        refused = {**REFUSED_PROBE_SECTION, "raises": "ValueError"}

        with pytest.raises(
            BotCriticalError,
            match=r"^_ProbeStrategy: profile 2 \(ETH/USDT:USDT 1h\): probe refusal$",
        ):
            _create(_ProbeStrategy, [VALID_PROBE, refused], TradingMode.BACKTEST)

    @pytest.mark.parametrize(
        "changes",
        [changes for changes, _ in UNBUILDABLE_SECTIONS.values()],
        ids=list(UNBUILDABLE_SECTIONS),
    )
    def test_live_with_every_section_unbuildable_stops(self, changes):
        refused = {**REFUSED_PROBE_SECTION, **changes}

        with pytest.raises(BotCriticalError, match="every profile was refused"):
            _create(_ProbeStrategy, [refused], TradingMode.LIVE)

    @pytest.mark.parametrize(
        "section",
        [
            {"symbl": "ETH/USDT:USDT", "timeframe": "1h", "quantity": 1},
            {"symbol": "ETHUSDT", "timeframe": "1h", "quantity": 1},
        ],
        ids=["key misspelt", "symbol that does not parse"],
    )
    def test_live_stops_on_a_section_whose_symbol_cannot_be_read(self, section):
        with pytest.raises(
            BotCriticalError, match=r"^_ProbeStrategy: profile 2 \(.*\): .*no symbol"
        ):
            _create(_ProbeStrategy, [VALID_PROBE, section], TradingMode.LIVE)

    def test_live_takes_every_profile_of_a_refused_symbol_out_of_the_run(self, caplog):
        strategy = _refused_on_btc_strategy()

        assert [str(profile.symbol) for profile in strategy.profiles] == [
            "SOL/USDT:USDT"
        ]
        assert sitting_out_symbols(strategy) == {Symbol.create("BTC/USDT:USDT")}
        assert [record.getMessage() for record in caplog.records] == [
            "_CrossingStrategy: profile 2 (BTC/USDT:USDT 1h): fast (50) must be "
            "below slow (30); this profile sits out the run",
            "_CrossingStrategy: profile 1 (BTC/USDT:USDT 4h): left out of the run "
            "with profile 2, which was refused on the same symbol",
        ]

    def test_live_with_every_profile_on_a_refused_symbol_stops(self):
        with pytest.raises(BotCriticalError, match="left out with a refused"):
            _create(
                _CrossingStrategy,
                [ACCEPTED_CROSSING, REFUSED_CROSSING_ON_BTC],
                TradingMode.LIVE,
            )

    def test_the_symbols_of_the_refused_profiles_are_named(self):
        strategy = _create(
            _CrossingStrategy,
            [REFUSED_CROSSING, ACCEPTED_CROSSING],
            TradingMode.LIVE,
        )

        assert sitting_out_symbols(strategy) == {Symbol.create("ETH/USDT:USDT")}


class TestRefusalsAsFailures:
    def test_a_profile_left_out_with_a_refused_one_has_no_failure_of_its_own(self):
        strategy = _refused_on_btc_strategy()

        with collect_profile_failures(strategy) as failures:
            assert outcomes(failures) == [(BTC, False)]

    def test_a_refused_section_is_one_failure_of_its_symbol_with_the_text_logged(
        self, caplog
    ):
        strategy = _create(
            _CrossingStrategy, [REFUSED_CROSSING, ACCEPTED_CROSSING], TradingMode.LIVE
        )

        with collect_profile_failures(strategy) as (failure,):
            assert failure.symbol == ETH
            assert not failure.recoverable
            assert failure.message == caplog.records[0].getMessage()

    def test_a_strategy_that_is_no_profile_strategy_has_no_failure(self):
        with collect_profile_failures(Mock(spec=StrategyProtocol)) as failures:
            assert failures == []

    def test_two_refused_sections_of_one_symbol_sit_out_one_symbol_and_fail_twice(
        self,
    ):
        strategy = _create(
            _CrossingStrategy,
            [REFUSED_CROSSING, ACCEPTED_CROSSING, REFUSED_CROSSING],
            TradingMode.LIVE,
        )

        assert sitting_out_symbols(strategy) == {ETH}
        with collect_profile_failures(strategy) as failures:
            assert len(failures) == 2


class TestDryRunSlots:
    def test_a_block_left_by_an_exception_puts_the_strategy_back_to_the_live_handling(
        self, caplog
    ):
        with pytest.raises(ExchangeCriticalError):
            with self._collected(signal_error=ValueError("boom")) as (
                strategy,
                failures,
            ):
                raise ExchangeCriticalError("down")

        _assert_generation_logs_one_error(strategy, failures, caplog)

    def test_a_critical_error_of_a_slot_stops_the_bot_once_the_dry_run_is_over(self):
        error = BotCriticalError("broken")
        with self._collected(signal_error=error) as (strategy, failures):
            pass

        with pytest.raises(BotCriticalError):
            strategy.generate_trading_signals(object())

        assert failures == []

    def test_a_failure_of_a_slot_is_logged_at_error_once_the_dry_run_is_over(
        self, caplog
    ):
        with self._collected(signal_error=ValueError("boom")) as (strategy, failures):
            pass

        _assert_generation_logs_one_error(strategy, failures, caplog)

    @pytest.mark.parametrize(
        ("slot_error", "signalled"),
        [("booking_error", [BTC, ETH]), ("signal_error", [ETH])],
    )
    def test_a_slot_failure_is_one_failure_of_the_symbol_and_the_other_runs(
        self, slot_error, signalled
    ):
        with self._collected(**{slot_error: ValueError("boom")}) as (
            strategy,
            failures,
        ):
            strategy.generate_trading_signals(object())
            _book(strategy, ["4h"])

        assert outcomes(failures) == [(BTC, False)]
        assert strategy.signalled == signalled
        assert strategy.booked == [ETH]

    def test_the_returned_list_keeps_growing_with_the_strategy(self):
        with self._collected(signal_error=ValueError("boom")) as (strategy, failures):
            strategy.generate_trading_signals(object())
            strategy.generate_trading_signals(object())

        assert len(failures) == 2

    @contextmanager
    def _collected(
        self, **settings: Any
    ) -> Iterator[tuple[_SymbolStrategy, list[DryRunFailure]]]:
        strategy = _create(
            _SymbolStrategy,
            [ACCEPTED_CROSSING, ACCEPTED_CROSSING_ON_ETH],
            TradingMode.LIVE,
            failing_symbol=BTC,
            **settings,
        )
        with collect_profile_failures(strategy) as failures:
            yield strategy, failures


class TestSettings:
    def test_a_key_set_on_its_attribute_by_type(self):
        strategy = _create(
            _TypedStrategy,
            [SECTION],
            TradingMode.BACKTEST,
            lookback="7",
            entry_order="limit",
        )

        assert strategy.lookback == 7
        assert strategy.entry_order == "limit"

    def test_the_default_when_the_key_is_absent(self):
        strategy = _create(_TypedStrategy, [SECTION], TradingMode.BACKTEST, lookback=5)

        assert strategy.entry_order == "trigger"

    def test_a_value_the_attribute_type_refuses(self):
        with pytest.raises(
            BotCriticalError,
            match=r"\[strategy\] key entry_order: Input should be 'trigger' or 'limit'",
        ):
            _create(
                _TypedStrategy,
                [SECTION],
                TradingMode.BACKTEST,
                lookback=5,
                entry_order="stop",
            )

    def test_a_key_naming_no_attribute(self):
        with pytest.raises(
            BotCriticalError,
            match=r"unknown \[strategy\] key lookbak; it declares entry_order, lookback",
        ):
            _create(
                _TypedStrategy, [SECTION], TradingMode.BACKTEST, lookback=5, lookbak=3
            )

    def test_a_key_on_a_strategy_declaring_none(self):
        with pytest.raises(
            BotCriticalError, match=r"unknown \[strategy\] key fast_length$"
        ):
            _create(_BareStrategy, PROFILES, TradingMode.BACKTEST, fast_length=20)

    def test_an_attribute_without_a_default_left_out(self):
        with pytest.raises(
            BotCriticalError, match=r"\[strategy\] key lookback is required"
        ):
            _create(_TypedStrategy, [SECTION], TradingMode.BACKTEST)

    def test_a_class_variable_is_no_setting(self):
        with pytest.raises(
            BotCriticalError, match=r"unknown \[strategy\] key market_type"
        ):
            _create(
                _TypedStrategy,
                [SECTION],
                TradingMode.BACKTEST,
                lookback=5,
                market_type="spot",
            )


class TestProfileSizing:
    @pytest.mark.parametrize(
        ("key", "value", "expected_rule"),
        [
            ("total_balance_ratio", 0.15, TotalBalanceRatio(0.15)),
            ("available_balance_ratio", 0.15, AvailableBalanceRatio(0.15)),
            ("equity_ratio", 0.15, EquityRatio(0.15)),
            ("risk_ratio", 0.02, RiskRatio(0.02)),
            (
                "risk_ratio",
                {"ratio": 0.02, "of": "total_balance"},
                RiskRatio(0.02, of="total_balance"),
            ),
            ("margin", 40, Margin(40.0)),
            ("notional", 200, Notional(200.0)),
            ("quantity", 3, Quantity(3.0)),
        ],
    )
    def test_an_engine_rule_built_from_its_key(self, key, value, expected_rule):
        section = {**SECTION, key: value}

        strategy = _create(_SizedStrategy, [section], TradingMode.BACKTEST)

        (profile,) = strategy.profiles
        assert profile.sizing == expected_rule

    def test_a_rule_the_strategy_registers_from_a_scalar(self):
        section = {**SECTION, "my_risk_ratio": 0.03}

        strategy = _create(_SizedStrategy, [section], TradingMode.BACKTEST)

        (profile,) = strategy.profiles
        assert profile.sizing == _MyRiskRatio(0.03)

    def test_a_rule_the_strategy_registers_from_a_table(self):
        section = {**SECTION, "my_kelly": {"lookback": 100}}

        strategy = _create(_SizedStrategy, [section], TradingMode.BACKTEST)

        (profile,) = strategy.profiles
        assert profile.sizing == _MyKelly(lookback=100)

    def test_a_profile_naming_no_rule(self):
        with pytest.raises(
            BotCriticalError,
            match=r"profile 1 \(BTC/USDT:USDT 4h\): names no sizing rule; add one of "
            r"available_balance_ratio, equity_ratio, margin, my_kelly, my_risk_ratio",
        ):
            _create(_SizedStrategy, [SECTION], TradingMode.BACKTEST)

    def test_a_profile_naming_two_rules(self):
        section = {**SECTION, "total_balance_ratio": 0.15, "notional": 200}

        with pytest.raises(
            BotCriticalError,
            match=r"profile 1 \(BTC/USDT:USDT 4h\): names several sizing rules, "
            r"total_balance_ratio, notional",
        ):
            _create(_SizedStrategy, [section], TradingMode.BACKTEST)

    def test_a_value_the_rule_refuses(self):
        section = {**SECTION, "total_balance_ratio": 1.5}

        with pytest.raises(
            BotCriticalError,
            match=r"profile 1 \(.*\): total_balance_ratio: `ratio` must be above 0.0",
        ):
            _create(_SizedStrategy, [section], TradingMode.BACKTEST)

    def test_a_margin_that_is_not_positive(self):
        section = {**SECTION, "margin": 0}

        with pytest.raises(
            BotCriticalError,
            match=r"profile 1 \(BTC/USDT:USDT 4h\): margin: `margin` must be greater than 0",
        ):
            _create(_SizedStrategy, [section], TradingMode.BACKTEST)

    def test_a_table_key_the_rule_does_not_take(self):
        section = {**SECTION, "risk_ratio": {"ratio": 0.02, "off": "equity"}}

        with pytest.raises(
            BotCriticalError, match=r"risk_ratio: .*unexpected keyword argument"
        ):
            _create(_SizedStrategy, [section], TradingMode.BACKTEST)

    def test_a_rule_registered_under_the_key_of_an_engine_rule(self):
        with pytest.raises(TypeError, match="sizing rule notional is the engine's"):

            class _Shadowing(ProfileStrategy[_SizedProfile]):
                sizing_rules = {"notional": _MyRiskRatio}

    def test_a_profile_declaring_no_sizing_keeps_the_key_its_own(self):
        section = {**SECTION, "notional": 200}

        with pytest.raises(
            BotCriticalError, match=r"profile 1 \(.*\): notional: unknown key"
        ):
            _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)


class TestStrategySizing:
    def test_a_rule_built_from_the_strategy_keys(self):
        strategy = _create(
            _StrategySizedOnce, [SECTION], TradingMode.BACKTEST, notional=200
        )

        assert strategy.sizing == Notional(200.0)

    def test_a_strategy_naming_no_rule(self):
        with pytest.raises(
            BotCriticalError,
            match=r"_StrategySizedOnce: \[strategy\] names no sizing rule",
        ):
            _create(_StrategySizedOnce, [SECTION], TradingMode.BACKTEST)

    def test_live_a_strategy_naming_no_rule_stops_the_bot_whole(self):
        with pytest.raises(
            BotCriticalError,
            match=r"_StrategySizedOnce: \[strategy\] names no sizing rule",
        ):
            _create(_StrategySizedOnce, [SECTION], TradingMode.LIVE)

    def test_live_a_strategy_value_its_rule_refuses_stops_the_bot_whole(self):
        with pytest.raises(
            BotCriticalError,
            match=r"_StrategySizedOnce: \[strategy\] notional: `notional` must be",
        ):
            _create(_StrategySizedOnce, [SECTION], TradingMode.LIVE, notional=-1)


class TestDefaultSlots:
    def test_a_strategy_overriding_no_slot(self):
        strategy = _create(_BareStrategy, PROFILES, TradingMode.BACKTEST)
        bookkeeper = BookKeeper()

        strategy.generate_trading_signals(object())
        strategy.book_trading_actions(object(), object(), TIMESTAMP, bookkeeper, ["1d"])

        assert bookkeeper.list_actions() == []


class TestClassDefinition:
    def test_a_misspelt_slot(self):
        with pytest.raises(TypeError, match="did you mean `book_profile_actions`"):

            class _Misspelt(ProfileStrategy[_Profile]):
                def book_profile_action(
                    self, profile, ohlcvs, account_snapshots, timestamp, bookkeeper
                ):
                    pass

    def test_an_override_of_a_framework_owned_method(self):
        with pytest.raises(TypeError, match="belongs to the framework"):

            class _Overriding(ProfileStrategy[_Profile]):
                def book_trading_actions(self, *args):
                    pass

    def test_a_slot_with_the_wrong_number_of_parameters(self):
        with pytest.raises(TypeError, match="book_profile_actions takes 1"):

            class _Wrong(ProfileStrategy[_Profile]):
                def book_profile_actions(self, profile):
                    pass

    def test_a_slot_taking_its_parameters_as_args(self):
        class _Splat(ProfileStrategy[_Profile]):
            def book_profile_actions(self, *args):
                pass

    def test_a_subclass_without_the_brackets(self):
        with pytest.raises(TypeError, match=r"ProfileStrategy\[MyProfile\]"):

            class _Unnamed(ProfileStrategy):
                pass

    def test_a_subclass_naming_a_type_variable(self):
        with pytest.raises(TypeError, match="names no profile class"):

            class _Generic[T: _Profile](ProfileStrategy[T]):
                pass

    def test_private_helpers_and_setup(self):
        class _Helpers(ProfileStrategy[_Profile]):
            async def setup(self, requirements):
                pass

            def _book_profile_rungs(self, profile):
                pass


def _dry_run(strategy: ProfileStrategy[Any]) -> None:
    with collect_profile_failures(strategy):
        pass


class TestProfileChecks:
    @pytest.mark.parametrize("entry", [_generate, _dry_run], ids=["run", "dry run"])
    def test_an_initialiser_that_never_calls_the_base(self, entry):
        with pytest.raises(BotCriticalError, match=r"must call super\(\)\.__init__"):
            entry(_Unset())

    def test_a_profile_lacking_a_field(self):
        strategy = _create(
            _NoTimeframeStrategy, [{"profile_id": "alpha"}], TradingMode.BACKTEST
        )

        with pytest.raises(BotCriticalError, match="lacks timeframe"):
            strategy.generate_trading_signals(object())

    def test_profiles_sharing_a_profile_id(self):
        profiles = [
            {"profile_id": "alpha", "timeframe": "1d"},
            {"profile_id": "alpha", "timeframe": "4h"},
        ]
        strategy = _create(_BareStrategy, profiles, TradingMode.BACKTEST)

        with pytest.raises(BotCriticalError, match="share a profile_id: alpha"):
            strategy.generate_trading_signals(object())


class TestSlotLogging:
    def test_the_slots_a_strategy_fills(self, caplog):
        profiles = [
            {"profile_id": "alpha", "timeframe": "1d"},
            {"profile_id": "beta", "timeframe": "1d"},
            {"profile_id": "gamma", "timeframe": "1d"},
        ]
        strategy = _create(_ThreeSlotStrategy, profiles, TradingMode.LIVE)

        with caplog.at_level(logging.DEBUG):
            log_strategy_slots(strategy)

        assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
            (logging.INFO, "_ThreeSlotStrategy: 3 profiles"),
            (
                logging.DEBUG,
                "_ThreeSlotStrategy slots: per-profile signals, "
                "general booking, per-profile booking",
            ),
        ]

    def test_the_slots_stay_out_of_an_info_log(self, caplog):
        strategy = _create(_ThreeSlotStrategy, PROFILES[:1], TradingMode.LIVE)

        with caplog.at_level(logging.INFO):
            log_strategy_slots(strategy)

        assert caplog.messages == ["_ThreeSlotStrategy: 1 profile"]

    def test_a_strategy_filling_no_slot(self, caplog):
        strategy = _create(_BareStrategy, PROFILES[:1], TradingMode.BACKTEST)

        with caplog.at_level(logging.DEBUG):
            log_strategy_slots(strategy)

        assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
            (logging.INFO, "_BareStrategy: 1 profile"),
            (logging.DEBUG, "_BareStrategy slots: none"),
        ]

    def test_a_strategy_that_does_not_run_profile_by_profile(self, caplog):
        with caplog.at_level(logging.INFO):
            log_strategy_slots(Mock(spec=StrategyProtocol))

        assert caplog.messages == []

    def test_a_strategy_that_never_called_the_base_initialiser(self, caplog):
        with caplog.at_level(logging.INFO):
            with pytest.raises(BotCriticalError, match=r"super\(\)\.__init__"):
                log_strategy_slots(_Unset())

        assert caplog.messages == []


def _assert_generation_logs_one_error(
    strategy: ProfileStrategy[Any],
    failures: list[DryRunFailure],
    caplog: pytest.LogCaptureFixture,
) -> None:
    strategy.generate_trading_signals(object())

    assert failures == []
    assert [record.levelno for record in caplog.records] == [logging.ERROR]
