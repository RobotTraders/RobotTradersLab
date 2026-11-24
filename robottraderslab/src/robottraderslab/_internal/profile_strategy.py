import difflib
import inspect
import logging
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
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

from robottraderslab_core import Symbol, TimeFrame
from robottraderslab_core.engine import AccountProtocol, sizing_rules
from robottraderslab_core.exceptions import BotCriticalError, ProfileConfigError
from robottraderslab_core.strategies import (
    AccountSnapshots,
    BookKeeper,
    OHLCVs,
    SizingRule,
    StrategyProtocol,
)

from .dry_run_failure import DryRunFailure
from .profile_failure_boundary import (
    ProfileSlot,
    dry_run_boundary,
    profile_failure_boundary,
)
from .profile_protocol import ProfileProtocol
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

_refusal_sink: ContextVar[list[DryRunFailure] | None] = ContextVar(
    "refusal_sink", default=None
)


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
    writes no `__init__`. Profiles run in declared order.

    Live, a failure one profile owns reaches that profile alone. A section
    that cannot be built is refused, whatever the cause, and every profile on
    its symbol sits out the run, since the profiles of one symbol share its
    position, its protections and its tracker: the symbol is left as it
    stands, and the run stops only when no profile is left. A bot with a
    working copy of its last working configuration file runs the symbol from
    the copy in exit-only mode instead when the section came with an edit. A
    profile class refuses its own section by raising `ProfileConfigError` from
    its `__post_init__`. An `Exception` inside a profile's signal or booking slot
    takes the profile out of the cycle, and a profile whose signal slot did not
    run through books nothing on that cycle. A `BotCriticalError` and an
    `ExchangeCriticalError` stop the bot wherever they are raised, a profile
    class's `__post_init__` and a profile's slot included, except that a dry
    run returns as a failure a `BotCriticalError` a slot or a step of the run
    raises, and a bot with a working copy runs the copy when the dry run of an
    edited file meets one. The general and final slots run outside that
    isolation: an `Exception` there costs every profile the cycle (signals) or
    what the candle had not yet booked (booking). In a backtest every failure
    stops the run.

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
        """Build the profiles from their sections and set the strategy's settings.

        Args:
            account: The account the run trades through.
            trading_system: Its trading mode decides whether a refused section
                takes its symbol out of the run (live, or runs it from the working
                copy after an edit) or stops it.
            config_dir: The directory the configuration was read from, None for
                a configuration built in code.
            profiles: The `[[strategy.profiles]]` sections. Each built
                profile needs a `profile_id` unique among them and a
                `timeframe`, checked before the first candle.
            settings: The remaining `[strategy]` keys, each set on the class
                attribute of the same name.

        Raises:
            BotCriticalError: If a `[strategy]` key names no attribute the
                class declares, holds a value its type refuses or leaves out a
                setting that has no default, or if a `sizing` the strategy
                declares is named by no key or by several; if, outside live
                trading, a section holds a key the profile class has no field
                for, a value its type refuses or a `sizing` named by no key or
                by several; if live trading leaves no profile to run, or a
                refused section holds no symbol that can be read; if a profile
                class or a sizing rule raises one while a section is built.
            ProfileConfigError: If the profile class refuses a section and
                the run is not live trading.
        """
        self.account = account
        self.config_dir = config_dir
        self.__trading_mode = trading_system.trading_mode
        self.profiles, self._refusals = _build_profiles(
            type(self), self.__trading_mode, profiles
        )
        self._failures: list[DryRunFailure] | None = None
        self._unsignalled: set[str] = set()
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
        if self.__trading_mode is TradingMode.LIVE:
            self._book_live(
                ohlcvs, account_snapshots, timestamp, bookkeeper, triggered_timeframes
            )
        else:
            for profile in self.profiles:
                if profile.timeframe in triggered_timeframes:
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
        """Compute the general signals, then every profile's, each isolated.

        A profile whose signal slot did not run through books nothing on this
        candle, and in a dry run its failure is reported to the caller.

        Args:
            ohlcvs: The candles the signal columns are computed on and stored in.

        Raises:
            BotCriticalError: If the profiles fail the checks of a first run,
                or a slot raises it; a profile's slot in a dry run excepted,
                which reports it.
            ExchangeCriticalError: If a slot raises it, in every mode.
            Exception: If the general slot raises one, or in a backtest a
                profile's slot does; live, a profile's slot isolates its own.
        """
        self._check_profiles()
        self.generate_general_signals(ohlcvs)
        if self.__trading_mode is not TradingMode.LIVE:
            for profile in self.profiles:
                self.generate_profile_signals(profile, ohlcvs)
            return
        self._unsignalled.clear()
        for profile in self.profiles:
            if not self._run_slot(
                ProfileSlot.SIGNALS,
                profile,
                self.generate_profile_signals,
                profile,
                ohlcvs,
            ):
                self._unsignalled.add(profile.profile_id)

    def _book_live(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: list[TimeFrame],
    ) -> None:
        for profile in self.profiles:
            if (
                profile.timeframe in triggered_timeframes
                and profile.profile_id not in self._unsignalled
            ):
                self._run_slot(
                    ProfileSlot.BOOKING,
                    profile,
                    self.book_profile_actions,
                    profile,
                    ohlcvs,
                    account_snapshots,
                    timestamp,
                    bookkeeper,
                )

    def _check_initialised(self) -> None:
        """Refuse an instance whose subclass never called the base initialiser.

        Raises:
            BotCriticalError: If the subclass's initialiser never called the
                base's.
        """
        if not hasattr(self, "_ProfileStrategy__trading_mode"):
            raise BotCriticalError(
                f"{type(self).__name__}.__init__ must call super().__init__(**kwargs)"
            )

    def _check_profiles(self) -> None:
        """Check the instance the subclass initialised before running it.

        Raises:
            BotCriticalError: If the subclass's initialiser never called
                the base's, a profile lacks `profile_id` or `timeframe`, or two
                profiles share a profile id.
        """
        self._check_initialised()
        name = type(self).__name__
        profiles = self.profiles
        for profile in profiles:
            missing = [
                field for field in _PROFILE_FIELDS if not hasattr(profile, field)
            ]
            if missing:
                raise BotCriticalError(
                    f"{name}: profile {profile!r} lacks {', '.join(missing)}"
                )
        counts = Counter(str(profile.profile_id) for profile in profiles)
        duplicated = sorted(
            profile_id for profile_id, count in counts.items() if count > 1
        )
        if duplicated:
            raise BotCriticalError(
                f"{name}: profiles share a profile_id: {', '.join(duplicated)}"
            )

    def _run_slot(
        self,
        slot: ProfileSlot,
        profile: TProfile,
        operation: Callable[..., None],
        *arguments: Any,
    ) -> bool:
        """Run one profile's slot behind the boundary the strategy reports to.

        Returns:
            False when a failure was contained, so work that depends on the
            slot having run through is left undone.
        """
        if self._failures is None:
            return profile_failure_boundary(
                self.__trading_mode, slot, profile.profile_id, operation, *arguments
            )
        return dry_run_boundary(self._failures, slot, profile, operation, *arguments)


def log_strategy_slots(strategy: StrategyProtocol) -> None:
    """Log a profile strategy's profile count at INFO and its filled slots at DEBUG.

    The INFO line reads `<Strategy>: <n> profile(s)`. The DEBUG line reads
    `<Strategy> slots: <labels>`, or `<Strategy> slots: none` when the class
    fills no slot.

    Raises:
        BotCriticalError: If the strategy's profiles fail the checks its
            first run would make.
    """
    if not isinstance(strategy, ProfileStrategy):
        return
    strategy._check_profiles()
    profile_count = len(strategy.profiles)
    plural = "" if profile_count == 1 else "s"
    strategy_name = type(strategy).__name__
    logger.info("%s: %d profile%s", strategy_name, profile_count, plural)
    logger.debug(
        "%s slots: %s",
        strategy_name,
        ", ".join(_filled_slot_labels(type(strategy))) or "none",
    )


@contextmanager
def collect_profile_failures(
    strategy: StrategyProtocol,
) -> Iterator[list[DryRunFailure]]:
    """Report a live profile strategy's failures for the length of a dry run.

    Within the block an `Exception` or a `BotCriticalError` a profile's slot
    raises is appended to the yielded list in place of being logged,
    attributed to the profile's symbol. The list already holds one failure,
    not recoverable, for each section the build refused, which the build
    logged. Leaving the block puts the strategy back to the live handling,
    which contains no `BotCriticalError`.

    Yields:
        The list of failures, which keeps growing as the strategy runs; empty
        for a strategy that is no profile strategy.

    Raises:
        BotCriticalError: If the strategy's initialiser never called the
            base's.
        ExchangeCriticalError: If a slot raises it, which leaves the block as
            raised.
    """
    failures: list[DryRunFailure] = []
    if not isinstance(strategy, ProfileStrategy):
        yield failures
        return
    strategy._check_initialised()
    failures.extend(
        DryRunFailure(symbol, message, recoverable=False)
        for symbol, message in strategy._refusals
    )
    strategy._failures = failures
    try:
        yield failures
    finally:
        strategy._failures = None


@contextmanager
def collecting_refusals() -> Iterator[list[DryRunFailure]]:
    """Collect every section a live build refuses for the length of a block.

    A dry run opens it around all its steps, so the refusals the build met are
    at hand when a later step stops the run before any strategy can report
    them.

    Yields:
        The list of failures, one for each refused section in the order the
        build refused them, naming its symbol, which keeps growing while the
        block runs.
    """
    sink: list[DryRunFailure] = []
    token = _refusal_sink.set(sink)
    try:
        yield sink
    finally:
        _refusal_sink.reset(token)


def sitting_out_symbols(strategy: StrategyProtocol) -> set[Symbol]:
    """The symbols a refused section took out of the run, which the cleanup of
    removed profiles must leave as they stand.
    """
    if not isinstance(strategy, ProfileStrategy):
        return set()
    return {symbol for symbol, _ in strategy._refusals}


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


def _build_profiles(
    cls: type["ProfileStrategy[Any]"],
    trading_mode: TradingMode,
    sections: Iterable[dict[str, Any]],
) -> tuple[list[Any], list[tuple[Symbol, str]]]:
    """Build every section into its profile.

    Live, a refused section takes its whole symbol out of the run: the
    profiles of one symbol share its net position, its protections and its
    tracker, which none of them owns, so no profile on it runs while one is
    refused. Outside live a refusal stops the run.

    Returns:
        The profiles that run, in declared order, and each refusal as the
        symbol it takes out of the run with the message the build logged.

    Raises:
        BotCriticalError: If live trading leaves no profile to run, or a
            refused section holds no symbol that can be read, or a profile
            class or a sizing rule raises one while a section is built.
    """
    section_list = list(sections)
    built: list[tuple[int, Any]] = []
    refused_at: dict[Symbol, int] = {}
    refusals: list[tuple[Symbol, str]] = []
    for index, section in enumerate(section_list, start=1):
        try:
            built.append((index, _build_profile(cls, index, section, trading_mode)))
        except ProfileConfigError as refusal:
            if trading_mode is not TradingMode.LIVE:
                raise
            symbol = _refused_symbol(refusal, section)
            refused_at.setdefault(symbol, index)
            message = f"{refusal}; this profile sits out the run"
            refusals.append((symbol, message))
            logger.error(message)
            _collect_refusal(symbol, message)
    if not refused_at:
        return [profile for _, profile in built], []
    profiles = _beyond_refused_symbols(cls, built, section_list, refused_at)
    if not profiles:
        raise BotCriticalError(
            f"{cls.__name__}: every profile was refused or left out with a refused "
            "profile's symbol, so no profile is left to run; correct the profile "
            "sections the ERROR lines above name in the bot's configuration"
        )
    return profiles, refusals


def _collect_refusal(symbol: Symbol, message: str) -> None:
    sink = _refusal_sink.get()
    if sink is not None:
        sink.append(DryRunFailure(symbol, message, recoverable=False))


def _beyond_refused_symbols(
    cls: type["ProfileStrategy[Any]"],
    built: list[tuple[int, Any]],
    sections: list[dict[str, Any]],
    refused_at: dict[Symbol, int],
) -> list[Any]:
    profiles = []
    for index, profile in built:
        symbol: Any = getattr(profile, "symbol", None)
        refused_index = refused_at.get(symbol)
        if refused_index is None:
            profiles.append(profile)
            continue
        logger.error(
            "%s: left out of the run with profile %d, which was refused on the same "
            "symbol",
            _where(cls, index, sections[index - 1]),
            refused_index,
        )
    return profiles


def _build_profile(
    cls: type["ProfileStrategy[Any]"],
    index: int,
    section: dict[str, Any],
    trading_mode: TradingMode,
) -> Any:
    """Build one section into its profile.

    Live, a section that cannot be built refuses its profile, whatever the
    `Exception` that says so. A `BotCriticalError` raised while the section is
    built, by the profile class or by a sizing rule, stops the bot, live and in
    a backtest, except that a bot with a working copy runs the copy when the
    dry run of an edited file meets one. Outside live the cause keeps its
    class: a section the engine cannot read is a `BotCriticalError`, and an
    exception raised by `__post_init__` propagates as raised.

    Raises:
        ProfileConfigError: If the profile class refuses the section, or,
            live, the section cannot be built.
        BotCriticalError: If, outside live, the section holds a key the
            profile class has no field for, a value its type refuses or a
            `sizing` named by no key or by several.
    """
    where = _where(cls, index, section)
    live = trading_mode is TradingMode.LIVE
    unbuildable = ProfileConfigError if live else BotCriticalError
    try:
        if cls._profile_sizes:
            try:
                section = _with_sizing(section, cls.sizing_rules)
            except ProfileConfigError as error:
                if live:
                    raise
                raise BotCriticalError(f"{where}: {error}") from error
        return cls._profile_adapter.validate_python(section)
    except ValidationError as error:
        reasons = "; ".join(
            f"{key}: {reason}" if key else reason for key, reason in _reasons(error)
        )
        raise unbuildable(f"{where}: {reasons}") from error
    except ProfileConfigError as error:
        raise ProfileConfigError(f"{where}: {error}") from error
    except Exception as error:
        if not live:
            raise
        raise ProfileConfigError(f"{where}: {type(error).__name__}: {error}") from error


def _refused_symbol(refusal: ProfileConfigError, section: dict[str, Any]) -> Symbol:
    """
    Raises:
        BotCriticalError: If the section holds no symbol that can be read, so
            that no market can be named for the profile to sit out.
    """
    try:
        return Symbol.create(section["symbol"])
    except (KeyError, TypeError, ValueError) as error:
        raise BotCriticalError(
            f"{refusal}; the section has no symbol that can be read, so the market "
            "it sits out cannot be named"
        ) from error


def _apply_settings(strategy: Any, settings: dict[str, Any]) -> None:
    """Set every `[strategy]` key on the typed class attribute it names.

    Raises:
        BotCriticalError: If a key names no attribute, a value is refused
            by the attribute's type, or an attribute without a default is
            not set.
    """
    cls = type(strategy)
    declared = _declared_settings(cls)
    if _SIZING_FIELD in declared:
        try:
            settings = _with_sizing(settings, cls.sizing_rules)
        except ProfileConfigError as error:
            raise BotCriticalError(f"{cls.__name__}: [strategy] {error}") from error
    unknown = sorted(set(settings) - set(declared))
    if unknown:
        known = f"; it declares {', '.join(declared)}" if declared else ""
        raise BotCriticalError(
            f"{cls.__name__}: unknown [strategy] key {', '.join(unknown)}{known}"
        )
    for name, hint in declared.items():
        if name in settings:
            try:
                value = TypeAdapter(hint).validate_python(settings[name])
            except ValidationError as error:
                reasons = "; ".join(reason for _, reason in _reasons(error))
                raise BotCriticalError(
                    f"{cls.__name__}: [strategy] key {name}: {reasons}"
                ) from error
            setattr(strategy, name, value)
        elif not hasattr(cls, name):
            raise BotCriticalError(f"{cls.__name__}: [strategy] key {name} is required")


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
        ProfileConfigError: If no key or several keys name a rule, or the
            rule refuses its value.
    """
    rule = SizingRule.from_settings(section, rules)
    registry = sizing_rules(rules)
    kept = {key: value for key, value in section.items() if key not in registry}
    return {**kept, _SIZING_FIELD: rule}


def _where(cls: type, index: int, section: dict[str, Any]) -> str:
    return f"{cls.__name__}: profile {index}{_market_of(section)}"


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
