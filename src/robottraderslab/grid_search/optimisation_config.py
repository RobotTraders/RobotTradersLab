import os
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from robottraderslab.exceptions import BotCriticalError

from .parameter import (
    ExplicitParameter,
    Parameter,
    ParameterValue,
    RangeParameter,
    SamplingType,
    _ValueType,
)


class ParameterConfig(BaseModel):
    """Configuration for a single parameter to optimise."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parameter: str
    sampling: SamplingType
    min: float | None = None
    max: float | None = None
    num_samples: int | None = None
    step: float | None = None
    values: list[ParameterValue] | None = None


class ProfileSelector(BaseModel):
    """The keys that single out the profile of `bot.toml` a search sweeps.

    A profile is known by its identity, never by its place in the list, so a
    reordered `bot.toml` sweeps the same profile.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str | None = None
    timeframe: str | None = None
    tag: str | None = None

    @model_validator(mode="after")
    def _names_a_key(self) -> "ProfileSelector":
        if not self._given():
            raise ValueError("`[profile]` gives none of `symbol`, `timeframe`, `tag`")
        return self

    def index_in(self, profiles: Sequence[Mapping[str, Any]]) -> int:
        """Find the one profile every key given agrees with.

        Raises:
            BotCriticalError: If no profile matches, or more than one does.
        """
        given = self._given()
        matches = [
            index
            for index, profile in enumerate(profiles)
            if all(profile.get(key) == value for key, value in given.items())
        ]
        if len(matches) == 1:
            return matches[0]
        outcome = "matches no profile" if not matches else "matches more than one"
        held = "; ".join(_identity(profile) for profile in profiles) or "none"
        raise BotCriticalError(
            f"`[profile]` of `grid-search.toml` ({_identity(given)}) {outcome} "
            f"of `bot.toml`, which holds: {held}"
        )

    def _given(self) -> dict[str, str]:
        return self.model_dump(exclude_none=True)


def _identity(profile: Mapping[str, Any]) -> str:
    keys = ("symbol", "timeframe", "tag")
    return ", ".join(f"{key} = {profile[key]}" for key in keys if key in profile)


_BOT_SECTIONS = ("strategy", "backtest", "live", "report", "orders")


class OptimisationConfig(BaseModel):
    """What a grid search declares: the profile it sweeps, its parameters,
    their ranges and its settings, and nothing else of the bot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile: ProfileSelector
    parameter_configs: list[ParameterConfig]
    max_workers: int = Field(default_factory=lambda: os.cpu_count() or 1, ge=1)

    @classmethod
    def from_file(cls, search_file: str | Path) -> "OptimisationConfig":
        """Read the search a `grid-search.toml` declares.

        Args:
            search_file: The TOML file declaring the search.

        Raises:
            BotCriticalError: If the file is not there, is not valid TOML,
                carries a section of the bot, which belongs in `bot.toml`, or
                declares what `from_config` refuses.
            UnicodeDecodeError: If the file is not valid UTF-8.
            OSError: If the file exists but cannot be read.
        """
        path = Path(search_file)
        if not path.is_file():
            raise BotCriticalError(f"There is no grid search file at `{path}`")
        try:
            declared = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise BotCriticalError(f"`{path}` is not valid TOML: {e}") from e
        carried = [f"[{name}]" for name in _BOT_SECTIONS if name in declared]
        if carried:
            raise BotCriticalError(
                f"`{path.name}` holds the search alone and carries "
                f"{', '.join(carried)}; the bot belongs in `bot.toml`"
            )
        return cls.from_config(declared)

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> "OptimisationConfig":
        """Validate what a grid search declared.

        Args:
            config_dict: The keys the grid search file declares.

        Raises:
            BotCriticalError: If the section declares a key the sweep
                does not read, leaves out one it needs, or gives a value it
                cannot take.
        """
        try:
            return cls(**config_dict)
        except ValidationError as e:
            raise BotCriticalError(
                f"The grid search is not one the engine can run: {e}"
            ) from e

    def to_parameters(
        self, current_values: Mapping[str, ParameterValue]
    ) -> list[Parameter]:
        """Build the parameter each declaration calls for.

        An integer setting samples whole numbers and a float setting floats
        with nothing declared: a declaration carries no type of its own.

        Args:
            current_values: The value the bot's config holds at each declared
                path.

        Returns:
            One parameter per declaration.
        """
        return [
            _parameter_of(pc, type(current_values[pc.parameter]))
            for pc in self.parameter_configs
        ]


def _parameter_of(pc: ParameterConfig, value_type: _ValueType) -> Parameter:
    if pc.sampling == SamplingType.EXPLICIT:
        return _explicit_parameter(pc, value_type)
    return _range_parameter(pc, value_type)


def _explicit_parameter(pc: ParameterConfig, value_type: _ValueType) -> Parameter:
    if pc.values is None:
        raise BotCriticalError(
            f"Parameter '{pc.parameter}': explicit sampling requires values"
        )
    return ExplicitParameter(
        name=pc.parameter, values=tuple(pc.values), value_type=value_type
    )


def _range_parameter(pc: ParameterConfig, value_type: _ValueType) -> Parameter:
    if pc.min is None or pc.max is None:
        raise BotCriticalError(
            f"Parameter '{pc.parameter}': {pc.sampling} sampling requires min and max"
        )
    if pc.num_samples is None and pc.step is None:
        raise BotCriticalError(
            f"Parameter '{pc.parameter}': {pc.sampling} sampling requires "
            f"num_samples or step"
        )
    return RangeParameter(
        name=pc.parameter,
        sampling=pc.sampling,
        min_value=pc.min,
        max_value=pc.max,
        num_samples=pc.num_samples,
        step=pc.step,
        value_type=value_type,
    )
