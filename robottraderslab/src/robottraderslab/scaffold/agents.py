import hashlib
import json
from collections.abc import Iterable
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from .report import report_written

_SHIPPED = files("robottraderslab") / "agents"
_INSTRUCTIONS = "AGENTS.md"
_SKILLS = "skills"
_AGENTS_DIR = Path("agents")
MANIFEST_NAME = ".robottraderslab.json"
_EDITED = "Edited since rtlab init wrote it:"
_ADVICE = (
    "An edit of yours belongs in your own instructions (CLAUDE.md, or the file your agent reads) or in a skill of your own; "
    "rerun with --force to replace these copies."
)


class EditedCopiesError(Exception):
    """Raised when a rerun would overwrite copies the user edited."""


def lay_out_agent_files(project: Path, force: bool) -> tuple[list[Path], list[Path]]:
    """So any AI agent can read the files where they are or take them into its
    own layout.

    Raises:
        EditedCopiesError: If an edited copy is found and force is not set.
    """
    folder = project / _AGENTS_DIR
    return copy_agent_files(
        project,
        folder / _INSTRUCTIONS,
        folder / _SKILLS,
        folder / MANIFEST_NAME,
        force=force,
    )


def copy_agent_files(
    project: Path, instructions: Path, skills: Path, manifest: Path, force: bool
) -> tuple[list[Path], list[Path]]:
    """The manifest is what tells a copy the user edited from one the engine
    wrote, so a later run never overwrites an edit silently.

    Returns:
        The paths written, and the paths removed because the manifest records
        them and the engine does not ship them.

    Raises:
        EditedCopiesError: If a copy was edited and `force` is not set, or the
            manifest cannot be read; nothing is written.
    """
    shipped = _shipped_files(instructions, skills)
    recorded = _read_manifest(project, manifest)
    stale = _stale(project, recorded, shipped)
    if not force:
        _refuse_edited(project, recorded, [*shipped, *stale])
    for path in stale:
        _remove(path)
    written = [_copy(source, target) for target, source in shipped.items()]
    _write_manifest(project, manifest, written)
    return written, stale


def main(project: Path, force: bool) -> None:
    written, removed = lay_out_agent_files(project, force)
    report_agent_files(written, removed)


def report_agent_files(written: Iterable[Path], removed: Iterable[Path]) -> None:
    report_written(written)
    for path in removed:
        print(f"Removed {path}")


def _shipped_files(instructions: Path, skills: Path) -> dict[Path, Traversable]:
    shipped = {instructions: _SHIPPED / _INSTRUCTIONS}
    for skill in (_SHIPPED / _SKILLS).iterdir():
        for file in skill.iterdir():
            shipped[skills / skill.name / file.name] = file
    return shipped


def _read_manifest(project: Path, manifest: Path) -> dict[str, str]:
    if not manifest.is_file():
        return {}
    try:
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
    except UnicodeDecodeError as e:
        raise _unreadable(manifest, "is not UTF-8 text") from e
    except json.JSONDecodeError as e:
        raise _unreadable(manifest, "is not valid JSON") from e
    if not isinstance(recorded, dict) or not all(
        isinstance(name, str) and isinstance(digest, str)
        for name, digest in recorded.items()
    ):
        raise _unreadable(manifest, "does not map paths to hashes")
    root = project.resolve()
    for name in recorded:
        if not (project / name).resolve().is_relative_to(root):
            raise _unreadable(manifest, f"names {name}, outside the project")
    return recorded


def _unreadable(manifest: Path, reason: str) -> EditedCopiesError:
    return EditedCopiesError(
        f"{manifest} {reason}; delete it and rerun to record the copies afresh"
    )


def _stale(
    project: Path, recorded: dict[str, str], shipped: dict[Path, Traversable]
) -> list[Path]:
    return [
        project / name for name in sorted(recorded) if project / name not in shipped
    ]


def _refuse_edited(
    project: Path, recorded: dict[str, str], paths: Iterable[Path]
) -> None:
    edited = []
    for path in paths:
        name = _name(project, path)
        if name in recorded and path.is_file() and _digest(path) != recorded[name]:
            edited.append(path)
    if edited:
        lines = [f"{_EDITED} {path}" for path in edited]
        raise EditedCopiesError("\n".join([*lines, _ADVICE]))


def _name(project: Path, path: Path) -> str:
    return path.relative_to(project).as_posix()


def _digest(path: Path) -> str:
    """Git on Windows checks a text file out with CRLF endings, so a copy cloned
    there is the same copy as the one written.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _remove(path: Path) -> None:
    path.unlink(missing_ok=True)
    if path.parent.is_dir() and not any(path.parent.iterdir()):
        path.parent.rmdir()


def _copy(source: Traversable, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return target


def _write_manifest(project: Path, manifest: Path, written: Iterable[Path]) -> None:
    recorded = {_name(project, path): _digest(path) for path in written}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(recorded, indent=2, sort_keys=True), encoding="utf-8"
    )
