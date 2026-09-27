from dataclasses import dataclass

from .parameter import ParameterValue


@dataclass(frozen=True, slots=True)
class GridPoint:
    """A single point in the parameter grid."""

    parameters: dict[str, ParameterValue]

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"GridPoint({params_str})"
