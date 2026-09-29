from pathlib import Path

_PROJECT_MARKER = "pyproject.toml"


class OutsideProjectError(Exception):
    """Raised when no folder at or above the start holds a `pyproject.toml`."""


def project_folder(start: Path) -> Path:
    """The project is the one `uv run` runs the engine of, so what a command
    lays out lands in it wherever inside it the command is run from.

    Raises:
        OutsideProjectError: If no folder at or above `start` holds a
            `pyproject.toml`.
    """
    for folder in (start, *start.parents):
        if (folder / _PROJECT_MARKER).is_file():
            return folder
    raise OutsideProjectError(
        f"no `{_PROJECT_MARKER}` at or above `{start}`; run the command from the "
        "project folder"
    )
