import inspect
from collections.abc import Iterable, Mapping


def settings_the_factory_does_not_take(
    declared: Mapping[str, inspect.Parameter], settings: Iterable[str]
) -> list[str]:
    """Name the configured settings a plugin factory has no parameter for.

    A factory taking arbitrary keyword arguments applies the ones it names
    and swallows the rest, so a setting meant for another plugin, or a
    mistyped name, would otherwise pass for applied. A factory that declares
    nothing but a catch-all states no contract to check, and one that
    declares no catch-all refuses an unknown setting on its own.
    """
    kinds = [parameter.kind for parameter in declared.values()]
    if inspect.Parameter.VAR_KEYWORD not in kinds:
        return []
    if all(kind is inspect.Parameter.VAR_KEYWORD for kind in kinds):
        return []
    return sorted(setting for setting in settings if setting not in declared)
