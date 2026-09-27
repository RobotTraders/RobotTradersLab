from pathlib import Path

_TOOL_DIR_NAME = ".rtlab"


def tool_file(owner: Path, suffix: str) -> Path:
    """Two bots named alike in two folders never share one file."""
    resolved = owner.resolve()
    root = _root_above(resolved)
    return root / resolved.relative_to(root.parent).with_suffix(suffix)


def tool_root(owner: Path) -> Path:
    """An operator places `.rtlab` once, at whatever level of a workspace should
    gather the files of everything under it.
    """
    return _root_above(owner.resolve())


def mark_root(directory: Path) -> None:
    """The marker gathers the files of everything under the directory holding
    it, so a workspace is one root however many bots it grows.
    """
    (directory / _TOOL_DIR_NAME).mkdir(parents=True, exist_ok=True)


def _root_above(resolved: Path) -> Path:
    for directory in resolved.parents:
        candidate = directory / _TOOL_DIR_NAME
        if candidate.is_dir():
            return candidate
    return resolved.parent / _TOOL_DIR_NAME
