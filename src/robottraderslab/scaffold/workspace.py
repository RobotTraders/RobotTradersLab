from importlib.resources import files
from pathlib import Path

from robottraderslab._core import mark_root

_TEMPLATES_DIR_NAME = "templates"
_EXAMPLE_FILES = ("secrets.example.toml", "registry.example.toml")
_GITIGNORE_TEMPLATE = "gitignore"
_GITIGNORE_NAME = ".gitignore"


def lay_out_workspace(root: Path) -> list[Path]:
    """The examples are the engine's, so a run over a workspace that already
    holds them writes the current ones.

    Args:
        root: The workspace directory.
    """
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in _EXAMPLE_FILES:
        target = root / name
        target.write_text(_template(name), encoding="utf-8")
        written.append(target)
    mark_workspace(root)
    return written


def mark_workspace(root: Path) -> None:
    """The workspace's `.gitignore` keeps what the engine writes and the
    secrets file out of a commit, and is written once so the user's own lines
    survive every later scaffold.
    """
    mark_root(root)
    gitignore = root / _GITIGNORE_NAME
    if not gitignore.exists():
        gitignore.write_text(_template(_GITIGNORE_TEMPLATE), encoding="utf-8")


def main(workspace: Path) -> None:
    for path in lay_out_workspace(workspace):
        print(f"Wrote {path}")


def _template(name: str) -> str:
    template = files(__package__) / _TEMPLATES_DIR_NAME / name
    return template.read_text(encoding="utf-8")
