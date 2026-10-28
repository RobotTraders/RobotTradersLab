import shutil
from importlib.metadata import entry_points
from importlib.resources import as_file, files
from pathlib import Path

from .workspace import mark_workspace

_ENTRY_POINT_GROUP = "robot_traders_lab.examples"
_EXAMPLES_DIR_NAME = "examples"


class ExampleError(Exception):
    """Raised when an example cannot be listed or copied."""


def list_examples() -> list[str]:
    """Return the sorted names of the examples registered by installed packages."""
    return sorted(ep.name for ep in entry_points(group=_ENTRY_POINT_GROUP))


def copy_example(name: str, destination_root: str | Path, folder: str | Path) -> Path:
    """Every example copied into one destination shares that destination's
    engine files and its store of candles, which its marker gathers.

    Args:
        name: Name of a registered example.
        destination_root: The workspace that receives the bot's folder.
        folder: The bot's folder, relative to `destination_root`; its name
            is the bot's.

    Raises:
        ExampleError: If the example is not registered, its package ships
            no example files, or the destination already exists.
    """
    package = _resolve_package(name)

    source = files(package) / _EXAMPLES_DIR_NAME
    if not source.is_dir():
        raise ExampleError(f"Package '{package}' ships no example files")

    destination = Path(destination_root) / folder
    if destination.exists():
        raise ExampleError(f"'{destination}' already exists; not overwriting")

    with as_file(source) as source_path:
        shutil.copytree(source_path, destination)
    mark_workspace(Path(destination_root))
    return destination


def _resolve_package(name: str) -> str:
    eps = entry_points(group=_ENTRY_POINT_GROUP)
    for ep in eps:
        if ep.name == name:
            return ep.value
    available = ", ".join(sorted(ep.name for ep in eps)) or "none"
    raise ExampleError(
        f"No example named '{name}' is installed. Available: {available}"
    )
