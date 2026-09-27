import json
import logging
import tomllib
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from robottraderslab._core import (
    BASE_DELAY_SECONDS,
    DEFAULT_ANNUALIZATION_FACTOR,
    DEFAULT_CALCULATION_METHOD,
    DEFAULT_FEE_MODE,
    DEFAULT_RISK_FREE_RATE,
    MAX_ATTEMPTS,
    CalculationMethod,
    Currency,
    FeeMode,
    LogLevel,
    PlacementReserve,
    Symbol,
    TimeFrame,
    tool_root,
)
from robottraderslab.exceptions import StrategyCriticalError

from ..ohlcv_sources import keeps_candles

type AccountConfig = dict[str, Any]

logger = logging.getLogger(__name__)


class LoggingConfig(BaseModel):
    """Base logging configuration."""

    console_level: LogLevel = LogLevel.INFO
    file_level: LogLevel = LogLevel.INFO
    enable_console: bool = False
    enable_file: bool = True
    logfile: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_custom_handlers(cls, values: Any) -> Any:
        if isinstance(values, dict) and "custom_handlers" in values:
            raise ValueError(
                "`logging.custom_handlers` is not read; declare the handler "
                "under `notifier.<plugin>.log` instead"
            )
        return values


class BacktestLoggingConfig(LoggingConfig):
    """Backtest-specific logging configuration."""


class LiveLoggingConfig(LoggingConfig):
    """Live trading logging configuration."""

    retention_days: int = 60


_MOVED_TO_REPORT = ("reference_symbol", "reports_dir")


class BacktestConfig(BaseModel):
    """Backtest execution configuration."""

    notifier: dict[str, dict[str, dict[str, Any]]] = {}
    initial_balance: dict[Currency, float]
    maker_fee_rate: float
    taker_fee_rate: float
    fee_mode: FeeMode = DEFAULT_FEE_MODE
    placement_reserve: PlacementReserve = PlacementReserve()
    start_date: str
    end_date: str
    ohlcv_provider: dict[str, Any]
    logging: BacktestLoggingConfig = BacktestLoggingConfig()

    @model_validator(mode="before")
    @classmethod
    def _reject_report_keys(cls, values: Any) -> Any:
        if isinstance(values, dict):
            moved = [key for key in _MOVED_TO_REPORT if key in values]
            if moved:
                named = ", ".join(f"`backtest.{key}`" for key in moved)
                raise ValueError(
                    f"{named} is not read; declare it under `[report]` instead"
                )
        return values

    @model_validator(mode="after")
    def _refuse_inline_credentials(self) -> "BacktestConfig":
        refuse_credentials_in_notifiers("backtest.notifier", self.notifier)
        return self


class RetryConfig(BaseModel):
    """Retry parameters for transient exchange errors."""

    max_attempts: int = MAX_ATTEMPTS
    base_delay_seconds: float = BASE_DELAY_SECONDS


class LiveConfig(BaseModel):
    """Live trading execution configuration."""

    trading_account: AccountConfig
    ohlcv_provider: dict[str, Any]
    retry: RetryConfig = RetryConfig()
    logging: LiveLoggingConfig = LiveLoggingConfig()
    notifier: dict[str, dict[str, dict[str, Any]]] = {}

    @model_validator(mode="before")
    @classmethod
    def _reject_notifiers(cls, values: Any) -> Any:
        if isinstance(values, dict) and "notifiers" in values:
            raise ValueError(
                "`live.notifiers` is not read; declare each subscription "
                "under `live.notifier.<plugin>.<occasion>` instead"
            )
        return values

    @model_validator(mode="after")
    def _refuse_inline_credentials(self) -> "LiveConfig":
        _refuse_bare_credential("live.trading_account", self.trading_account)
        refuse_credentials_in_notifiers("live.notifier", self.notifier)
        return self


class ReportConfig(BaseModel):
    """How a run's results are reported, whichever run produced them.

    Nothing in a simulation reads it, so a value declared here changes what a
    report states and never what a run does.
    """

    model_config = ConfigDict(extra="forbid")

    reference_symbol: str | None = None
    reference_timeframe: TimeFrame | None = None
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR
    calculation_method: CalculationMethod = DEFAULT_CALCULATION_METHOD
    reports_dir: Path | None = None

    @field_validator("reference_symbol")
    @classmethod
    def _refuse_an_unparsable_symbol(cls, value: str | None) -> str | None:
        """The analyser parses this text once the run it reports on is over,
        so a load is the cheapest moment to refuse one it cannot read.
        """
        if value is not None:
            Symbol.create(value)
        return value


class StrategyConfig(BaseModel):
    """What a strategy declares, its own parameters included.

    A strategy owns parameters the engine knows nothing about, so a key it does
    not read is carried through to the strategy.
    """

    model_config = ConfigDict(extra="allow")

    strategy_class: str
    profiles: list[dict[str, Any]] = []


class BotConfig(BaseSettings):
    """A bot's configuration file, parsed and validated.

    One file describes one bot, so a backtest and a live run take the same
    configuration and differ only in the sections that bot declares. A section
    a run needs is asked for, and a name the engine does not know is refused.
    """

    model_config = SettingsConfigDict(extra="forbid")

    secrets_file: str | None = None
    config_dir: Path | None = None
    config_file: Path | None = None
    strategy: StrategyConfig
    backtest: BacktestConfig | None = None
    live: LiveConfig | None = None
    report: ReportConfig = ReportConfig()
    optimisation: dict[str, Any] | None = None

    @classmethod
    def from_file(
        cls, config_file: str | Path, *, overlay: str | None = None
    ) -> "BotConfig":
        """Read the bot a config file declares.

        A path a bot writes is anchored to the file's own directory, so a config
        means the same thing whatever directory a run starts from. An overlay is
        merged section by section, so a caller varying one value restates
        nothing else. A bot naming no secrets file of its own inherits one found beside
        it, or one shared by every bot in its workspace.

        Args:
            config_file: TOML file declaring the bot.
            overlay: TOML adding to or replacing what the file declares.

        Raises:
            StrategyCriticalError: If the named file is not there, is not valid
                TOML, or declares something other than a bot the engine can run.
        """
        logger.debug(f"Loading config from `{config_file}`")
        if not Path(config_file).is_file():
            raise StrategyCriticalError(f"There is no config file at `{config_file}`")

        base_dir = Path(config_file).resolve().parent
        try:
            declared = TomlConfigSettingsSource(cls, toml_file=str(config_file))()
        except tomllib.TOMLDecodeError as e:
            raise StrategyCriticalError(
                f"the config at `{config_file}` is not valid TOML: {e}"
            ) from e
        _resolve_relative_paths(declared, base_dir)
        _default_storage(declared, Path(config_file))
        if declared.get("secrets_file") is None:
            discovered = _discover_secrets_file(base_dir)
            if discovered is not None:
                logger.debug(f"Discovered secrets file at `{discovered}`")
            declared["secrets_file"] = discovered
        declared["config_dir"] = base_dir
        declared["config_file"] = Path(config_file)

        if overlay is not None:
            declared = _deep_merge(declared, tomllib.loads(overlay))

        logger.debug("Config declares: %s", _DeclaredShape(declared))
        return cls._validated(declared, config_file)

    @classmethod
    def from_text(cls, toml_config: str) -> "BotConfig":
        """Read the bot a TOML declaration describes.

        Nothing anchors a relative path without a file to anchor it to, so a bot
        described this way states every path in full.

        Args:
            toml_config: TOML declaring the bot.

        Raises:
            StrategyCriticalError: If what it declares is not a bot the engine
                can run.
        """
        declared = tomllib.loads(toml_config)
        logger.debug("Config declares: %s", _DeclaredShape(declared))
        return cls._validated(declared, None)

    @classmethod
    def _validated(
        cls, declared: dict[str, Any], source: str | Path | None
    ) -> "BotConfig":
        try:
            return cls(**declared)
        except ValidationError as e:
            raise StrategyCriticalError(_configuration_failure(source, e)) from e

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Name the only source a config may be built from.

        The settings framework calls this to ask which sources may supply a
        value, and in what order. Returning the init source alone leaves no
        environment variable, dotenv file or secrets directory able to reach a
        load, so a config comes from what the caller passes and nothing else.
        """
        return (init_settings,)


_CREDENTIAL_KEYS = frozenset(
    {
        "api_key",
        "api_secret",
        "secret_key",
        "passphrase",
        "private_key",
        "webhook_url",
    }
)


def refuse_credentials_in_notifiers(
    notifier_label: str, notifier: dict[str, dict[str, dict[str, Any]]]
) -> None:
    """Refuse a credential spelled out in any occasion of a notifier table.

    Args:
        notifier_label: What the notifier table itself is called, e.g.
            `live.notifier` or `notifier` for one with no `live`/`backtest`
            section of its own.

    Raises:
        ValueError: If any occasion's config declares one of a fixed set of
            credential key names as a plain value.
    """
    for plugin_name, occasions in notifier.items():
        for occasion_name, config in occasions.items():
            _refuse_bare_credential(
                f"{notifier_label}.{plugin_name}.{occasion_name}", config
            )


def _refuse_bare_credential(block_label: str, config: dict[str, Any]) -> None:
    """Refuse a credential spelled out in a config block.

    Raises:
        ValueError: If `config` declares one of a fixed set of credential key
            names as a plain value.
    """
    bare_keys = sorted(_CREDENTIAL_KEYS.intersection(config))
    if bare_keys:
        named = ", ".join(f"`{block_label}.{key}`" for key in bare_keys)
        raise ValueError(
            f"{named} is a bare credential; declare it under `[[secrets]]` "
            "in the secrets file and reference it with `secret_name` instead"
        )


def _configuration_failure(
    config_file: str | Path | None, error: ValidationError
) -> str:
    """State which declaration is wrong, so the fix is the file and not the
    traceback.
    """
    faults = "; ".join(
        f"{'.'.join(str(part) for part in fault['loc'])}: {fault['msg']}"
        for fault in error.errors()
    )
    source = f"`{config_file}`" if config_file is not None else "the given config"
    return f"{source} is not a valid configuration: {faults}"


def resolve_path(bot_config: BotConfig, key: str) -> Path | None:
    """Resolve a config value to an absolute path against the config directory.

    Relative values are anchored to the config file's directory; absolute values
    are returned unchanged.

    Args:
        bot_config: What the bot declared, read for the value at `key`.
        key: Dotted key of the value to resolve, e.g. "strategy.model_dir".

    Returns:
        Absolute path to the value, or None if the key is missing or empty.
    """
    value = _get_dotted_value(bot_config, key)
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if bot_config.config_dir is None:
        return path
    return bot_config.config_dir / path


_RELATIVE_PATHS = (
    "secrets_file",
    "backtest.ohlcv_provider.file",
    "backtest.ohlcv_provider.storage.dir",
    "backtest.ohlcv_provider.storage_dir",
    "backtest.logging.logfile",
    "live.ohlcv_provider.file",
    "live.ohlcv_provider.storage.dir",
    "live.ohlcv_provider.storage_dir",
    "live.logging.logfile",
    "report.reports_dir",
)


def _resolve_relative_paths(data: dict[str, Any], base_dir: Path) -> None:
    """Anchor a declared path to the file that declared it.

    A bot config is read from wherever it sits, so a path written relative to it
    means the same thing whatever directory the process was started from.
    """
    for key_path in _RELATIVE_PATHS:
        value = _get_dotted_value(data, key_path)
        if value and not Path(value).is_absolute():
            _set_dotted_value(data, key_path, str(base_dir / value))


def _get_dotted_value(source: Any, dotted_key: str) -> Any:
    value = source
    for part in dotted_key.split("."):
        if isinstance(value, BaseModel):
            value = getattr(value, part, None)
        elif isinstance(value, dict):
            value = value.get(part)
        else:
            return None
        if value is None:
            return None
    return value


def _set_dotted_value(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    *parents, leaf = dotted_key.split(".")
    target = data
    for part in parents:
        target = target[part]
    target[leaf] = value


_DEFAULT_SECRETS_FILENAME = "secrets.toml"


def _discover_secrets_file(config_dir: Path) -> str | None:
    """Find the secrets file a bot named none of its own.

    A bot missing its own secrets file may share one with its siblings, so the
    search reaches one directory up before giving up.
    """
    for candidate_dir in (config_dir, config_dir.parent):
        candidate = candidate_dir / _DEFAULT_SECRETS_FILENAME
        if candidate.is_file():
            return str(candidate)
    return None


_PROVIDER_SECTIONS = ("backtest.ohlcv_provider", "live.ohlcv_provider")
_DECLARED_STORAGE = ("storage_dir", "storage")
_DEFAULT_STORE = Path("data") / "ohlcvs"


def _default_storage(data: dict[str, Any], config_file: Path) -> None:
    """Give a bot naming no store the one its workspace shares.

    Every bot under one workspace root reads the same store, so a market is
    downloaded once for all of them; a source reading a file of its own or
    generating its candles has nothing to store.
    """
    workspace = tool_root(config_file).parent
    for section in _PROVIDER_SECTIONS:
        provider = _get_dotted_value(data, section)
        if not isinstance(provider, dict):
            continue
        source = provider.get("ohlcv_provider")
        if source is None or not keeps_candles(source):
            continue
        if any(key in provider for key in _DECLARED_STORAGE):
            continue
        provider["storage_dir"] = str(workspace / _DEFAULT_STORE)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge overlay into base, recursing so a nested table keeps its siblings."""
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


class _DeclaredShape:
    """Name what a config holds without rendering any of it.

    A bot may write a credential straight into its file, so a line that renders
    a value can put one in a log file or a chat channel.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __str__(self) -> str:
        return json.dumps(
            {
                section: sorted(value) if isinstance(value, dict) else "declared"
                for section, value in self._data.items()
            },
            indent=2,
            default=str,
        )
