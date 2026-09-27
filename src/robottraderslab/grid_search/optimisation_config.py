import os
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from robottraderslab.exceptions import StrategyCriticalError

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


class OptimisationConfig(BaseModel):
    """Configuration for grid search optimisation from TOML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parameter_configs: list[ParameterConfig]
    max_workers: int = Field(default_factory=lambda: os.cpu_count() or 1, ge=1)

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> "OptimisationConfig":
        """Validate what a bot declared for its sweep.

        Args:
            config_dict: The `[optimisation]` section the bot declares.

        Raises:
            StrategyCriticalError: If the section declares a key the sweep
                does not read, leaves out one it needs, or gives a value it
                cannot take.
        """
        try:
            return cls(**config_dict)
        except ValidationError as e:
            raise StrategyCriticalError(
                f"`[optimisation]` is not a sweep the engine can run: {e}"
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
        raise StrategyCriticalError(
            f"Parameter '{pc.parameter}': explicit sampling requires values"
        )
    return ExplicitParameter(
        name=pc.parameter, values=tuple(pc.values), value_type=value_type
    )


def _range_parameter(pc: ParameterConfig, value_type: _ValueType) -> Parameter:
    if pc.min is None or pc.max is None:
        raise StrategyCriticalError(
            f"Parameter '{pc.parameter}': {pc.sampling} sampling requires min and max"
        )
    if pc.num_samples is None and pc.step is None:
        raise StrategyCriticalError(
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
