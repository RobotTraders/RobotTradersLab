import difflib
import inspect
import logging
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Generic,
    TypeVar,
    final,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import TypeAdapter, ValidationError
from pydantic.errors import PydanticSchemaGenerationError

from .account_requirements import AccountProtocol
from .account_snapshot import AccountSnapshots
from .actions import BookKeeper
from .exceptions import StrategyCriticalError
from .interfaces import StrategyProtocol
from .ohlcv.ohlcvs import OHLCVs
from .profile_failure_boundary import profile_failure_boundary
from .profile_protocol import ProfileProtocol
from .sizing import SizingRule, sizing_rules
from .timeframes import TimeFrame
from .trading_system import TradingMode, TradingSystem

logger = logging.getLogger(__name__)

TProfile = TypeVar("TProfile", bound=ProfileProtocol)

_RUN_METHODS = ("generate_trading_signals", "book_trading_actions")
_SLOT_LABELS = {
    "generate_general_signals": "general signals",
    "generate_profile_signals": "per-profile signals",
    "book_general_actions": "general booking",
    "book_profile_actions": "per-profile booking",
    "book_final_actions": "final booking",
}
_SLOTS = tuple(_SLOT_LABELS)
_PROFILE_FIELDS = ("profile_id", "timeframe")
_ENGINE_ATTRIBUTES = frozenset({"account", "config_dir", "profiles"})
_UNKNOWN_KEY_TYPES = frozenset({"extra_forbidden", "unexpected_keyword_argument"})
_SIZING_FIELD = "sizing"


class ProfileStrategy(Generic[TProfile]):
    """Runs a strategy profile by profile, so one profile's failure stays its own.

    A subclass names its profile class in the brackets,
    `ProfileStrategy[MyProfile]`, sets `market_type`, writes `setup`, and
    overrides the slots it needs; every slot does nothing by default, and
    `generate_trading_signals` and `book_trading_actions` belong to the
    engine. Each `[[strategy.profiles]]` section of the configuration is
    built into the profile class by the types of its fields, a string
    becoming the enum, the timeframe or the symbol the field is typed as.
    Every other `[strategy]` key names a typed class attribute of the
    strategy and sets it by that type, so a subclass declares a setting of
    its own as `entry_order: Literal["trigger", "limit"] = "trigger"` and
    writes no `__init__`. Profiles run in declared order, each isolated in
    live trading so a failing profile sits out the candle while its siblings
    trade; in a backtest every failure propagates. The general and final
    slots run outside that isolation, so a failure there costs the cycle.

    A profile class declaring `sizing: SizingRule` has it built from the one
    key of its section named after a rule, `risk_ratio = 0.02` or
    `risk_ratio = { ratio = 0.02, of = "total_balance" }`, and a strategy
    declaring the attribute has it built the same way from its `[strategy]`
    keys. The keys are the engine's rules and those of `sizing_rules`.

    Attributes:
        account: The account the run trades through.
        config_dir: The directory the configuration was read from, None for a
            configuration built in code.
        profiles: The profiles built from the configuration, in declared order.
        sizing_rules: The strategy's own sizing rules by the key a
            configuration names each under, `{"my_kelly": MyKelly}`.
    """

    sizing_rules: ClassVar[Mapping[str, type[SizingRule]]] = MappingProxyType({})

    _profile_class: type[TProfile]
    _profile_adapter: TypeAdapter[TProfile]
    _profile_sizes: bool

    def __init__(
        self,
        *,
        account: AccountProtocol,
        trading_system: TradingSystem,
        config_dir: Path | None,
        profiles: Iterable[dict[str, Any]],
        **settings: Any,
    ) -> None:
        """
        Args:
            profiles: The `[[strategy.profiles]]` sections. Each built
                profile needs a `profile_id` unique among them and a
                `timeframe`, checked before the first candle.
            settings: The remaining `[strategy]` keys, each set on the class
                attribute of the same name.

        Raises:
            StrategyCriticalError: If a section or a setting holds a key the
                class declares no field or attribute for, a value its type
                refuses, or leaves out a setting that has no default, or if a
                `sizing` it declares is named by no key or by several.
        """
        self.account = account
        self.config_dir = config_dir
        self.__trading_mode = trading_system.trading_mode
        self.profiles = [
            _build_profile(type(self), index, section)
            for index, section in enumerate(profiles, start=1)
        ]
        _apply_settings(self, settings)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Raises:
            TypeError: If the class misspells a slot, overrides one with the
                wrong number of parameters, overrides a run method, names no
                profile class in its brackets, names one that cannot be built
                from a configuration section, or registers a sizing rule under
                the key of an engine rule.
        """
        super().__init_subclass__(**kwargs)
        _reject_stray_methods(cls)
        _reject_wrong_arity(cls)
        sizing_rules(cls.sizing_rules)
        cls._profile_class = _profile_class_of(cls)
        cls._profile_adapter = _profile_adapter_of(cls._profile_class)
        cls._profile_sizes = _SIZING_FIELD in get_type_hints(cls._profile_class)

    def book_final_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: list[TimeFrame],
    ) -> None:
        """Book what needs every profile's decision, after all profiles booked."""

    def book_general_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: list[TimeFrame],
    ) -> None:
        """Book what spans profiles, before any profile books.

        Work batched over symbols belongs here, where it runs once per
        candle.
        """

    def book_profile_actions(
        self,
        profile: TProfile,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
    ) -> None:
        """Book one profile's actions for the candle."""

    @final
    def book_trading_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: list[TimeFrame],
    ) -> None:
        """Book the general actions, every triggered profile's, then the final ones."""
        self.book_general_actions(
            ohlcvs, account_snapshots, timestamp, bookkeeper, triggered_timeframes
        )
        for profile in self.profiles:
            if profile.timeframe not in triggered_timeframes:
                continue
            if self.__trading_mode is TradingMode.LIVE:
                with profile_failure_boundary(
                    self.__trading_mode, str(profile.profile_id)
                ):
                    self.book_profile_actions(
                        profile, ohlcvs, account_snapshots, timestamp, bookkeeper
                    )
            else:
                self.book_profile_actions(
                    profile, ohlcvs, account_snapshots, timestamp, bookkeeper
                )
        self.book_final_actions(
            ohlcvs, account_snapshots, timestamp, bookkeeper, triggered_timeframes
        )

    def generate_general_signals(self, ohlcvs: OHLCVs) -> None:
        """Compute signals that span profiles, before any profile's own."""

    def generate_profile_signals(self, profile: TProfile, ohlcvs: OHLCVs) -> None:
        """Compute and store one profile's signal columns."""

    @final
    def generate_trading_signals(self, ohlcvs: OHLCVs) -> None:
        """Compute the general signals, then every profile's, each isolated."""
        self._check_profiles()
        self.generate_general_signals(ohlcvs)
        for profile in self.profiles:
            if self.__trading_mode is TradingMode.LIVE:
                with profile_failure_boundary(
                    self.__trading_mode, str(profile.profile_id)
                ):
                    self.generate_profile_signals(profile, ohlcvs)
            else:
                self.generate_profile_signals(profile, ohlcvs)

    def _check_profiles(self) -> None:
        """Check the instance the subclass initialised before running it.

        Raises:
            StrategyCriticalError: If the subclass's initialiser never called
                the base's, a profile lacks `profile_id` or `timeframe`, or two
                profiles share a profile id.
        """
        name = type(self).__name__
        if not hasattr(self, "_ProfileStrategy__trading_mode"):
            raise StrategyCriticalError(
                f"{name}.__init__ must call super().__init__(**kwargs)"
            )
        profiles = self.profiles
        for profile in profiles:
            missing = [
                field for field in _PROFILE_FIELDS if not hasattr(profile, field)
            ]
            if missing:
                raise StrategyCriticalError(
                    f"{name}: profile {profile!r} lacks {', '.join(missing)}"
                )
        counts = Counter(str(profile.profile_id) for profile in profiles)
        duplicated = sorted(
            profile_id for profile_id, count in counts.items() if count > 1
        )
        if duplicated:
            raise StrategyCriticalError(
                f"{name}: profiles share a profile_id: {', '.join(duplicated)}"
            )


def log_strategy_slots(strategy: StrategyProtocol) -> None:
    """Every slot runs whether a subclass fills it or not, so this line is the
    operator's only view of which ones do.

    Raises:
        StrategyCriticalError: If the strategy's profiles fail the checks its
            first run would make.
    """
    if not isinstance(strategy, ProfileStrategy):
        return
    strategy._check_profiles()
    profile_count = len(strategy.profiles)
    plural = "" if profile_count == 1 else "s"
    logger.info(
        "%s: %d profile%s; slots: %s",
        type(strategy).__name__,
        profile_count,
        plural,
        ", ".join(_filled_slot_labels(type(strategy))) or "none",
    )


def _reject_stray_methods(cls: type) -> None:
    """Refuse a public method that overrides a run method or misspells a slot.

    Raises:
        TypeError: If the class defines a public `generate_*` or `book_*`
            method that is not a slot.
    """
    for name, value in vars(cls).items():
        if name.startswith("_") or not callable(value):
            continue
        if name in _RUN_METHODS:
            raise TypeError(
                f"{cls.__name__}.{name} belongs to the framework; "
                "implement the slots instead"
            )
        if name.startswith(("generate_", "book_")) and name not in _SLOTS:
            close = difflib.get_close_matches(name, _SLOTS, n=1)
            hint = f"; did you mean `{close[0]}`?" if close else ""
            raise TypeError(
                f"{cls.__name__}.{name} is not a ProfileStrategy slot{hint}"
            )


def _reject_wrong_arity(cls: type) -> None:
    """Refuse a slot override whose positional parameters differ in number.

    Raises:
        TypeError: If a slot the class defines takes a different number of
            positional parameters from the slot it overrides.
    """
    for slot in _SLOTS:
        override = vars(cls).get(slot)
        if override is None or not callable(override):
            continue
        expected = _positional_parameters(getattr(ProfileStrategy, slot))
        actual = _positional_parameters(override)
        if actual is None or expected is None or len(actual) == len(expected):
            continue
        raise TypeError(
            f"{cls.__name__}.{slot} takes {len(actual)} positional parameters "
            f"after self; the slot takes {len(expected)}: {', '.join(expected)}"
        )


def _positional_parameters(function: Any) -> list[str] | None:
    """Name the positional parameters after self, or None when *args takes any."""
    parameters = list(inspect.signature(function).parameters.values())[1:]
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters):
        return None
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    return [p.name for p in parameters if p.kind in positional]


def _profile_class_of(cls: type) -> type:
    """
    Raises:
        TypeError: If neither the class nor an ancestor names a profile class
            in the brackets.
    """
    for base in vars(cls).get("__orig_bases__", ()):
        if get_origin(base) is ProfileStrategy:
            (profile_class,) = get_args(base)
            if isinstance(profile_class, type):
                return profile_class
    inherited: type | None = getattr(cls, "_profile_class", None)
    if inherited is None:
        raise TypeError(
            f"{cls.__name__} names no profile class; declare it as "
            f"`class {cls.__name__}(ProfileStrategy[MyProfile])`"
        )
    return inherited


def _profile_adapter_of(profile_class: type) -> TypeAdapter[Any]:
    """
    Raises:
        TypeError: If the profile class declares no typed fields to build a
            section by.
    """
    try:
        return TypeAdapter(profile_class)
    except PydanticSchemaGenerationError as error:
        raise TypeError(
            f"{profile_class.__name__} cannot be built from a configuration "
            "section; declare it as a dataclass with typed fields"
        ) from error


def _build_profile(
    cls: type["ProfileStrategy[Any]"], index: int, section: dict[str, Any]
) -> Any:
    """
    Raises:
        StrategyCriticalError: If the section holds a key the profile class
            has no field for, or a value its type refuses.
    """
    where = f"{cls.__name__}: profile {index}{_market_of(section)}"
    try:
        if cls._profile_sizes:
            section = _with_sizing(section, cls.sizing_rules)
        return cls._profile_adapter.validate_python(section)
    except ValidationError as error:
        reasons = "; ".join(f"{key}: {reason}" for key, reason in _reasons(error))
        raise StrategyCriticalError(f"{where}: {reasons}") from error
    except StrategyCriticalError as error:
        raise StrategyCriticalError(f"{where}: {error}") from error


def _apply_settings(strategy: Any, settings: dict[str, Any]) -> None:
    """Set every `[strategy]` key on the typed class attribute it names.

    Raises:
        StrategyCriticalError: If a key names no attribute, a value is refused
            by the attribute's type, or an attribute without a default is
            not set.
    """
    cls = type(strategy)
    declared = _declared_settings(cls)
    if _SIZING_FIELD in declared:
        try:
            settings = _with_sizing(settings, cls.sizing_rules)
        except StrategyCriticalError as error:
            raise StrategyCriticalError(
                f"{cls.__name__}: [strategy] {error}"
            ) from error
    unknown = sorted(set(settings) - set(declared))
    if unknown:
        known = f"; it declares {', '.join(declared)}" if declared else ""
        raise StrategyCriticalError(
            f"{cls.__name__}: unknown [strategy] key {', '.join(unknown)}{known}"
        )
    for name, hint in declared.items():
        if name in settings:
            try:
                value = TypeAdapter(hint).validate_python(settings[name])
            except ValidationError as error:
                reasons = "; ".join(reason for _, reason in _reasons(error))
                raise StrategyCriticalError(
                    f"{cls.__name__}: [strategy] key {name}: {reasons}"
                ) from error
            setattr(strategy, name, value)
        elif not hasattr(cls, name):
            raise StrategyCriticalError(
                f"{cls.__name__}: [strategy] key {name} is required"
            )


def _declared_settings(cls: type) -> dict[str, Any]:
    """The public annotated attributes of the class, the settings it declares."""
    return {
        name: hint
        for name, hint in get_type_hints(cls).items()
        if not name.startswith("_")
        and name not in _ENGINE_ATTRIBUTES
        and get_origin(hint) is not ClassVar
    }


def _with_sizing(
    section: dict[str, Any], rules: Mapping[str, type[SizingRule]]
) -> dict[str, Any]:
    """
    Raises:
        StrategyCriticalError: If no key or several keys name a rule, or the
            rule refuses its value.
    """
    rule = SizingRule.from_settings(section, rules)
    registry = sizing_rules(rules)
    kept = {key: value for key, value in section.items() if key not in registry}
    return {**kept, _SIZING_FIELD: rule}


def _market_of(section: dict[str, Any]) -> str:
    words = [str(section[key]) for key in ("symbol", "timeframe") if section.get(key)]
    return f" ({' '.join(words)})" if words else ""


def _reasons(error: ValidationError) -> list[tuple[str, str]]:
    """Each refused value as the key it sits under and why, in the error's order."""
    return [
        (".".join(str(part) for part in detail["loc"]), _reason(detail))
        for detail in error.errors()
    ]


def _reason(detail: Any) -> str:
    if detail["type"] in _UNKNOWN_KEY_TYPES:
        return "unknown key"
    if detail["type"] == "value_error":
        return str(detail["ctx"]["error"])
    return str(detail["msg"])


def _filled_slot_labels(strategy_class: type) -> list[str]:
    return [
        label
        for slot, label in _SLOT_LABELS.items()
        if getattr(strategy_class, slot) is not getattr(ProfileStrategy, slot)
    ]
