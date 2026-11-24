import importlib
from typing import Any

_CORE_GATE = importlib.import_module("robottraderslab_core.exchanges")

__all__ = list(_CORE_GATE.__all__)


def __getattr__(name: str) -> Any:
    """The names are the core's, resolved by its gate on first read, so
    a command that never reads one loads none of the candle libraries
    behind them.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_CORE_GATE, name)


def __dir__() -> list[str]:
    return __all__
