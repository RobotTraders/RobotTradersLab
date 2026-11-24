import importlib
from collections.abc import Callable
from functools import cache
from importlib.metadata import EntryPoint, entry_points
from types import ModuleType
from typing import Any


class ClassLoadingError(Exception):
    """Raised when a class cannot be loaded."""


def import_module(module_path: str) -> ModuleType:
    """Import a module by its dotted path.

    Args:
        module_path: Dotted module path (e.g., "robottraderslab_bitget.bitget_exchange")

    Raises:
        ClassLoadingError: If the module cannot be imported.
    """
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ClassLoadingError(
            f"Module `{module_path}` can't be imported; "
            "make sure to configure PYTHONPATH accordingly"
        ) from e


def get_class_from_module(module: ModuleType, class_name: str) -> Any:
    """Get a class from an imported module.

    Returns:
        The class object.

    Raises:
        ClassLoadingError: If the class doesn't exist in the module.
    """
    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise ClassLoadingError(
            f"There is no `{class_name}` class in module `{module.__name__}`"
        ) from e


def split_class_path(class_path: str) -> tuple[str, str]:
    """Split a full class path into module path and class name.

    Args:
        class_path: Full dotted path (e.g., "robottraderslab_bitget.bitget_exchange.BitgetExchange")

    Returns:
        Tuple of (module_path, class_name).

    Raises:
        ClassLoadingError: If the path doesn't contain a dot.
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        return module_path, class_name
    except ValueError as e:
        raise ClassLoadingError(
            f"Class path should be a dotted string (e.g. 'module.ClassName'); "
            f"got `{class_path}`"
        ) from e


def load_class(name: str, entry_point_group: str) -> Any:
    """Load a class by entry point name or full dotted path.

    Args:
        name: Entry point name (e.g., ``"simulator"``) or full dotted
            class path (e.g., ``"robottraderslab.backtester.simulator.create_simulator"``).
        entry_point_group: Entry point group for short name resolution.

    Returns:
        The loaded class.

    Raises:
        ClassLoadingError: If the class cannot be loaded.
    """
    if ":" in name:
        return _load_entry_point_value(name)

    if "." in name:
        module_path, class_name = split_class_path(name)
        module = import_module(module_path)
        return get_class_from_module(module, class_name)

    ep_value = _resolve_entry_point(name, entry_point_group)
    return _load_entry_point_value(ep_value)


def find_class(name: str, entry_point_group: str) -> Any | None:
    """Return what a name registered in the group resolves to, or None.

    A plugin registers a contribution of a kind only when it has one to make,
    so a name absent from the group means it has none.

    Args:
        name: Entry point name (e.g., ``"bitget"``), case-insensitive.

    Returns:
        What the entry point resolves to, or None when no installed
        distribution registers the name in the group.

    Raises:
        ClassLoadingError: If the name is registered but its entry point
            cannot be loaded.
    """
    registered = _entry_point_named(name, entry_point_group) is not None
    return load_class(name, entry_point_group) if registered else None


def registered_entry_point_name(name: str, entry_point_group: str) -> str | None:
    """Return the name the group registers for a configured class name.

    Args:
        name: A short name, matched case-insensitively, or a full path such
            as `module.attribute` or `module:attribute`, matched to the entry
            point that registers it.

    Returns:
        The registered name, or None when no installed distribution
        registers the class in the group.
    """
    if ":" in name or "." in name:
        value = _entry_point_value(name)
        entry_point = _first_entry_point(
            entry_point_group, lambda candidate: candidate.value == value
        )
    else:
        entry_point = _entry_point_named(name, entry_point_group)
    return None if entry_point is None else entry_point.name


def _load_entry_point_value(ep_value: str) -> Any:
    """Load a callable from an entry point value.

    Handles the standard ``module.path:Attr.chain`` format where the
    colon separates the module from a dotted attribute chain.

    Args:
        ep_value: Entry point value (e.g., ``"robottraderslab.backtester.simulator:create_simulator"``).

    Returns:
        The resolved attribute.

    Raises:
        ClassLoadingError: If the module or attribute cannot be loaded.
    """
    module_path, attr_chain = ep_value.split(":", 1)
    module = import_module(module_path)

    obj: Any = module
    for attr_name in attr_chain.split("."):
        try:
            obj = getattr(obj, attr_name)
        except AttributeError as e:
            raise ClassLoadingError(
                f"Attribute `{attr_name}` not found on `{obj}` "
                f"while resolving `{ep_value}`"
            ) from e
    return obj


@cache
def _resolve_entry_point(name: str, group: str) -> str:
    """Resolve a short name to an entry point value.

    Looking a group up reads the metadata of every installed distribution,
    and the answer holds for as long as the process runs, so each name is
    resolved once however many times it is asked for.

    Args:
        name: Short name to look up (e.g., "bitget"), case-insensitive.
        group: Entry point group (e.g., "robot_traders_lab.exchanges").

    Returns:
        Raw entry point value
        (e.g., ``"robottraderslab.backtester.simulator:create_simulator"``).

    Raises:
        ClassLoadingError: If the entry point is not found.
    """
    entry_point = _entry_point_named(name, group)
    if entry_point is None:
        raise ClassLoadingError(f"'{name}' not found in entry point group '{group}'")
    return entry_point.value


def _entry_point_named(name: str, group: str) -> EntryPoint | None:
    name_lower = name.lower()
    return _first_entry_point(
        group, lambda entry_point: entry_point.name.lower() == name_lower
    )


def _entry_point_value(class_path: str) -> str:
    return class_path if ":" in class_path else ":".join(split_class_path(class_path))


def _first_entry_point(
    group: str, matches: Callable[[EntryPoint], bool]
) -> EntryPoint | None:
    return next(
        (
            entry_point
            for entry_point in entry_points(group=group)
            if matches(entry_point)
        ),
        None,
    )
