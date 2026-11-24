from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import numpy as np


class SamplingType(StrEnum):
    """Type of parameter sampling strategy."""

    LINEAR = "linear"
    LOG = "log"
    EXPLICIT = "explicit"


type ParameterValue = int | float | str
type _ValueType = type[int] | type[float] | type[str]

_RANGEABLE_TYPES: tuple[_ValueType, ...] = (int, float)


@dataclass(frozen=True, slots=True)
class RangeParameter:
    """
    Parameter with range-based sampling (linear or logarithmic).

    Generates values between min_value and max_value using either
    num_samples (evenly spaced) or step (fixed increment, boundaries
    always included), of the type the setting holds in the bot's config.
    """

    name: str
    sampling: SamplingType
    value_type: _ValueType
    min_value: float
    max_value: float
    num_samples: int | None = None
    step: float | None = None

    def __post_init__(self) -> None:
        if self.sampling not in (SamplingType.LINEAR, SamplingType.LOG):
            raise ValueError(
                f"RangeParameter requires LINEAR or LOG sampling, got {self.sampling}"
            )
        if self.min_value >= self.max_value:
            raise ValueError(f"Parameter '{self.name}': min_value must be < max_value")

        has_step = self.step is not None
        has_num_samples = self.num_samples is not None
        if has_step == has_num_samples:
            raise ValueError(
                f"Parameter '{self.name}': provide exactly one of step or num_samples"
            )
        if self.step is not None:
            if self.sampling != SamplingType.LINEAR:
                raise ValueError(
                    f"Parameter '{self.name}': step is only supported with linear sampling"
                )
            if self.step <= 0:
                raise ValueError(f"Parameter '{self.name}': step must be > 0")
        if self.num_samples is not None and self.num_samples < 2:
            raise ValueError(
                f"Parameter '{self.name}': num_samples must be >= 2, got {self.num_samples}"
            )
        if self.sampling == SamplingType.LOG and self.min_value <= 0:
            raise ValueError(
                f"Parameter '{self.name}': log sampling requires min_value > 0"
            )
        if self.value_type not in _RANGEABLE_TYPES:
            raise ValueError(
                f"Parameter '{self.name}': range sampling needs a setting holding "
                "an integer or a float"
            )

    def generate_values(self) -> list[ParameterValue]:
        """Generate parameter values based on sampling strategy."""
        if self.step is not None:
            raw = self._generate_step_values()
        elif self.sampling == SamplingType.LINEAR:
            assert self.num_samples is not None
            raw = np.linspace(
                self.min_value,
                self.max_value,
                self.num_samples,
            ).tolist()
        else:  # LOG
            assert self.num_samples is not None
            raw = np.logspace(
                np.log10(self.min_value),
                np.log10(self.max_value),
                self.num_samples,
            ).tolist()

        if self.value_type is int:
            return [int(round(v)) for v in raw]
        return cast(list[ParameterValue], raw)

    def _generate_step_values(self) -> list[float]:
        assert self.step is not None
        n_steps = int((self.max_value - self.min_value) / self.step)
        values = [self.min_value + i * self.step for i in range(n_steps + 1)]
        if abs(values[-1] - self.max_value) > 1e-10:
            values.append(self.max_value)
        return values


@dataclass(frozen=True, slots=True)
class ExplicitParameter:
    """
    Parameter with explicit list of values to test.

    Uses exactly the values provided, cast to the type the setting holds in
    the bot's config.
    """

    name: str
    values: tuple[ParameterValue, ...]
    value_type: _ValueType

    def __post_init__(self) -> None:
        if len(self.values) < 2:
            raise ValueError(
                f"Parameter '{self.name}': explicit sampling requires at least 2 values"
            )

    def generate_values(self) -> list[ParameterValue]:
        return [self.value_type(v) for v in self.values]


# Union type for any parameter
Parameter = RangeParameter | ExplicitParameter
