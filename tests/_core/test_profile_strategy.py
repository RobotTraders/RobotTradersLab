import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal
from unittest.mock import Mock

import pytest

from robottraderslab import TimeFrame
from robottraderslab._core import log_strategy_slots
from robottraderslab.exceptions import ExchangeCriticalError, StrategyCriticalError
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
            raise StrategyCriticalError(f"length must be positive, got {self.length}")


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
        if self._failing and profile.profile_id == self._failing_profile_id:
            raise self._failing
        self.generated.append(profile.profile_id)

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
        if self._failing and profile.profile_id == self._failing_profile_id:
            raise self._failing
        self.booked.append(profile.profile_id)

    def book_final_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: Any,
        timestamp: datetime,
        bookkeeper: Any,
        triggered_timeframes: list[TimeFrame],
    ) -> None:
        self.booked.append("final")


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
            StrategyCriticalError,
            match=r"profile 1 \(BTC/USDT:USDT 4h\): pace: Input should be 'fast' or 'slow'",
        ):
            _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)

    def test_a_key_the_profile_has_no_field_for(self):
        section = {**SECTION, "lenght": 30}

        with pytest.raises(
            StrategyCriticalError, match=r"profile 1 \(.*\): lenght: unknown key"
        ):
            _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)

    def test_a_required_field_left_out(self):
        with pytest.raises(
            StrategyCriticalError, match=r"profile 2 \(4h\): symbol: Field required"
        ):
            _create(
                _TypedStrategy,
                [SECTION, {"timeframe": "4h"}],
                TradingMode.BACKTEST,
                lookback=5,
            )

    def test_the_profiles_own_rule(self):
        section = {**SECTION, "length": 0}

        with pytest.raises(
            StrategyCriticalError, match=r"profile 1 \(.*\): length must be positive"
        ):
            _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)

    def test_a_symbol_spelt_wrong(self):
        section = {**SECTION, "symbol": "BTCUSDT"}

        with pytest.raises(
            StrategyCriticalError, match="symbol: Invalid symbol format: BTCUSDT"
        ):
            _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)

    @pytest.mark.parametrize("tag", ["a@b", "a-b", "a/b", "a:b", "a b"])
    def test_a_tag_with_a_reserved_character(self, tag):
        section = {**SECTION, "tag": tag}

        with pytest.raises(
            StrategyCriticalError, match=f"tag: '{re.escape(tag)}' cannot contain"
        ):
            _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)

    def test_a_tag_over_eight_bytes(self):
        section = {**SECTION, "tag": "abcdefghi"}

        with pytest.raises(
            StrategyCriticalError,
            match=r"profile 1 \(.*\): tag: 'abcdefghi' is 9 bytes, over the 8-byte",
        ):
            _create(_TypedStrategy, [section], TradingMode.BACKTEST, lookback=5)


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
            StrategyCriticalError,
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
            StrategyCriticalError,
            match=r"unknown \[strategy\] key lookbak; it declares entry_order, lookback",
        ):
            _create(
                _TypedStrategy, [SECTION], TradingMode.BACKTEST, lookback=5, lookbak=3
            )

    def test_a_key_on_a_strategy_declaring_none(self):
        with pytest.raises(
            StrategyCriticalError, match=r"unknown \[strategy\] key fast_length$"
        ):
            _create(_BareStrategy, PROFILES, TradingMode.BACKTEST, fast_length=20)

    def test_an_attribute_without_a_default_left_out(self):
        with pytest.raises(
            StrategyCriticalError, match=r"\[strategy\] key lookback is required"
        ):
            _create(_TypedStrategy, [SECTION], TradingMode.BACKTEST)

    def test_a_class_variable_is_no_setting(self):
        with pytest.raises(
            StrategyCriticalError, match=r"unknown \[strategy\] key market_type"
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
            StrategyCriticalError,
            match=r"profile 1 \(BTC/USDT:USDT 4h\): names no sizing rule; add one of "
            r"available_balance_ratio, equity_ratio, margin, my_kelly, my_risk_ratio",
        ):
            _create(_SizedStrategy, [SECTION], TradingMode.BACKTEST)

    def test_a_profile_naming_two_rules(self):
        section = {**SECTION, "total_balance_ratio": 0.15, "notional": 200}

        with pytest.raises(
            StrategyCriticalError,
            match=r"profile 1 \(BTC/USDT:USDT 4h\): names several sizing rules, "
            r"total_balance_ratio, notional",
        ):
            _create(_SizedStrategy, [section], TradingMode.BACKTEST)

    def test_a_value_the_rule_refuses(self):
        section = {**SECTION, "total_balance_ratio": 1.5}

        with pytest.raises(
            StrategyCriticalError,
            match=r"profile 1 \(.*\): total_balance_ratio: `ratio` must be above 0.0",
        ):
            _create(_SizedStrategy, [section], TradingMode.BACKTEST)

    def test_a_margin_that_is_not_positive(self):
        section = {**SECTION, "margin": 0}

        with pytest.raises(
            StrategyCriticalError,
            match=r"profile 1 \(BTC/USDT:USDT 4h\): margin: `margin` must be greater than 0",
        ):
            _create(_SizedStrategy, [section], TradingMode.BACKTEST)

    def test_a_table_key_the_rule_does_not_take(self):
        section = {**SECTION, "risk_ratio": {"ratio": 0.02, "off": "equity"}}

        with pytest.raises(
            StrategyCriticalError, match=r"risk_ratio: .*unexpected keyword argument"
        ):
            _create(_SizedStrategy, [section], TradingMode.BACKTEST)

    def test_a_rule_registered_under_the_key_of_an_engine_rule(self):
        with pytest.raises(TypeError, match="sizing rule notional is the engine's"):

            class _Shadowing(ProfileStrategy[_SizedProfile]):
                sizing_rules = {"notional": _MyRiskRatio}

    def test_a_profile_declaring_no_sizing_keeps_the_key_its_own(self):
        section = {**SECTION, "notional": 200}

        with pytest.raises(
            StrategyCriticalError, match=r"profile 1 \(.*\): notional: unknown key"
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
            StrategyCriticalError,
            match=r"_StrategySizedOnce: \[strategy\] names no sizing rule",
        ):
            _create(_StrategySizedOnce, [SECTION], TradingMode.BACKTEST)


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


class TestProfileChecks:
    def test_an_initialiser_that_never_calls_the_base(self):
        class _Unset(ProfileStrategy[_Profile]):
            def __init__(self) -> None:
                self._profiles = PROFILES
                self._trading_mode = TradingMode.BACKTEST

        with pytest.raises(
            StrategyCriticalError, match=r"must call super\(\)\.__init__"
        ):
            _Unset().generate_trading_signals(object())

    def test_a_profile_lacking_a_field(self):
        strategy = _create(
            _NoTimeframeStrategy, [{"profile_id": "alpha"}], TradingMode.BACKTEST
        )

        with pytest.raises(StrategyCriticalError, match="lacks timeframe"):
            strategy.generate_trading_signals(object())

    def test_profiles_sharing_a_profile_id(self):
        profiles = [
            {"profile_id": "alpha", "timeframe": "1d"},
            {"profile_id": "alpha", "timeframe": "4h"},
        ]
        strategy = _create(_BareStrategy, profiles, TradingMode.BACKTEST)

        with pytest.raises(StrategyCriticalError, match="share a profile_id: alpha"):
            strategy.generate_trading_signals(object())


class TestSlotLogging:
    def test_the_slots_a_strategy_fills(self, caplog):
        profiles = [
            {"profile_id": "alpha", "timeframe": "1d"},
            {"profile_id": "beta", "timeframe": "1d"},
            {"profile_id": "gamma", "timeframe": "1d"},
        ]
        strategy = _create(_ThreeSlotStrategy, profiles, TradingMode.LIVE)

        with caplog.at_level(logging.INFO):
            log_strategy_slots(strategy)

        assert caplog.messages == [
            "_ThreeSlotStrategy: 3 profiles; slots: per-profile signals, "
            "general booking, per-profile booking"
        ]

    def test_a_strategy_filling_no_slot(self, caplog):
        strategy = _create(_BareStrategy, PROFILES[:1], TradingMode.BACKTEST)

        with caplog.at_level(logging.INFO):
            log_strategy_slots(strategy)

        assert caplog.messages == ["_BareStrategy: 1 profile; slots: none"]

    def test_a_strategy_that_does_not_run_profile_by_profile(self, caplog):
        with caplog.at_level(logging.INFO):
            log_strategy_slots(Mock(spec=StrategyProtocol))

        assert caplog.messages == []

    def test_a_strategy_that_never_called_the_base_initialiser(self, caplog):
        class _Unset(ProfileStrategy[_Profile]):
            def __init__(self) -> None:
                self._account = object()

        with caplog.at_level(logging.INFO):
            with pytest.raises(StrategyCriticalError, match=r"super\(\)\.__init__"):
                log_strategy_slots(_Unset())

        assert caplog.messages == []
