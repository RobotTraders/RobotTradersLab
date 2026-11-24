import logging
from typing import Any


def lowest_handler_level(handlers_config: dict[str, dict[str, Any]]) -> int:
    """Return the level the root logger should admit for these handlers.

    A record below every handler's level is built and then dropped, and
    building one is expensive: it walks the stack to find the caller. The root
    therefore admits no more than the most permissive handler asks for. A
    handler that declares no level of its own may consume anything, so it
    widens the root back to DEBUG.
    """
    if not handlers_config:
        return logging.DEBUG
    declared_levels: list[int | str] = []
    for config in handlers_config.values():
        if "level" not in config:
            return logging.DEBUG
        declared_levels.append(config["level"])
    return min(_severity(level) for level in declared_levels)


def _severity(level: int | str) -> int:
    if isinstance(level, int):
        return level
    return logging.getLevelNamesMapping()[str(level)]
