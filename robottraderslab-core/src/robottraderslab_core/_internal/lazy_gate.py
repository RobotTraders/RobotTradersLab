import importlib
from collections.abc import Callable, Mapping
from typing import Any


def resolve_on_read(
    gate: str, module_by_name: Mapping[str, str]
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """The `__getattr__` and `__dir__` of a gate resolving names on first read.

    `module_by_name` maps each name to the module that defines it, which is
    imported at the moment the name is read, so importing the gate loads no
    data library.
    """

    def get_name(name: str) -> Any:
        module = module_by_name.get(name)
        if module is None:
            raise AttributeError(f"module {gate!r} has no attribute {name!r}")
        return getattr(importlib.import_module(module), name)

    def list_names() -> list[str]:
        return sorted(module_by_name)

    return get_name, list_names
